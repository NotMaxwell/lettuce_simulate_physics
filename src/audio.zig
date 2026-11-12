const std = @import("std");
const c = @import("c");

pub const Sound = enum { hit, damage, death, explosion };

const sound_count = std.enums.values(Sound).len;
const voice_count = 8;
const repeat_window_ms: u64 = 40;
const music_path = "assets/sounds/fortnite-corail.mp3";

// Every clip is converted once at load so all streams share one source format
// and playback never has to renegotiate formats or resample.
const mix_spec: c.SDL_AudioSpec = .{ .format = c.SDL_AUDIO_F32, .channels = 2, .freq = 48000 };

pub const Info = struct { seconds: f32, sample_rate: u32, channels: u32 };

const Clip = struct {
    samples: []align(1) const u8 = &.{}, // SDL_malloc'd, f32 stereo @ 48 kHz
    info: Info = .{ .seconds = 0, .sample_rate = 0, .channels = 0 },

    fn load(path: [:0]const u8) Clip {
        var size: usize = 0;
        const file = c.SDL_LoadFile(path.ptr, &size) orelse return .{};
        defer c.SDL_free(file);

        var cfg: c.drmp3_config = undefined;
        var frames: c.drmp3_uint64 = 0;
        const pcm = c.drmp3_open_memory_and_read_pcm_frames_f32(file, size, &cfg, &frames, null) orelse return .{};
        defer c.drmp3_free(pcm, null);
        if (frames == 0 or cfg.channels == 0 or cfg.sampleRate == 0) return .{};

        const src: c.SDL_AudioSpec = .{ .format = c.SDL_AUDIO_F32, .channels = @intCast(cfg.channels), .freq = @intCast(cfg.sampleRate) };
        var out: ?[*]u8 = null;
        var out_len: c_int = 0;
        const in_len: c_int = @intCast(frames * cfg.channels * @sizeOf(f32));
        if (!c.SDL_ConvertAudioSamples(&src, @ptrCast(pcm), in_len, &mix_spec, @ptrCast(&out), &out_len)) return .{};
        return .{
            .samples = out.?[0..@intCast(out_len)],
            .info = .{
                .seconds = @as(f32, @floatFromInt(frames)) / @as(f32, @floatFromInt(cfg.sampleRate)),
                .sample_rate = cfg.sampleRate,
                .channels = cfg.channels,
            },
        };
    }

    fn free(self: *Clip) void {
        if (self.samples.len != 0) c.SDL_free(@constCast(self.samples.ptr));
        self.* = .{};
    }
};

const Voice = struct {
    stream: ?*c.SDL_AudioStream = null,
    sound: Sound = .hit,
    serial: u64 = 0,
};

const State = struct {
    gpa: std.mem.Allocator,
    device: c.SDL_AudioDeviceID = 0,
    clips: [sound_count]Clip = @splat(.{}),
    voices: [voice_count]Voice = @splat(.{}),
    last_played_ms: [sound_count]u64 = @splat(0),
    serial: u64 = 0,

    music_stream: ?*c.SDL_AudioStream = null,
    music_thread: ?*c.SDL_Thread = null,
    music: Clip = .{},
    // Release-stored by the decoder thread after `music` is filled; the audio
    // callback only reads `music` once it observes this.
    music_ready: std.atomic.Value(bool) = .init(false),
    music_pos: usize = 0, // audio thread only
};

fn clipPath(s: Sound) [:0]const u8 {
    return switch (s) {
        .hit => "assets/sounds/hitmarker.mp3",
        .damage => "assets/sounds/minecraft-damage.mp3",
        .death => "assets/sounds/minecraft-death.mp3",
        .explosion => "assets/sounds/minecraft-explosion.mp3",
    };
}

fn musicThread(data: ?*anyopaque) callconv(.c) c_int {
    const st: *State = @ptrCast(@alignCast(data.?));
    st.music = Clip.load(music_path);
    if (st.music.samples.len != 0) st.music_ready.store(true, .release);
    return 0;
}

fn musicCallback(userdata: ?*anyopaque, stream: ?*c.SDL_AudioStream, additional: c_int, total: c_int) callconv(.c) void {
    _ = total;
    const st: *State = @ptrCast(@alignCast(userdata.?));
    if (!st.music_ready.load(.acquire) or additional <= 0) return;
    const buf = st.music.samples;
    var left: usize = @intCast(additional);
    while (left > 0) {
        if (st.music_pos >= buf.len) st.music_pos = 0;
        const n = @min(left, buf.len - st.music_pos);
        if (!c.SDL_PutAudioStreamData(stream, buf.ptr + st.music_pos, @intCast(n))) return;
        st.music_pos += n;
        left -= n;
    }
}

pub const Audio = struct {
    state: ?*State,

    /// Never fails; without a device or files it degrades to silence.
    pub fn init(gpa: std.mem.Allocator) Audio {
        const st = gpa.create(State) catch return .{ .state = null };
        st.* = .{ .gpa = gpa };
        if (!c.SDL_InitSubSystem(c.SDL_INIT_AUDIO)) {
            gpa.destroy(st);
            return .{ .state = null };
        }
        st.device = c.SDL_OpenAudioDevice(c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &mix_spec);
        if (st.device == 0) {
            c.SDL_QuitSubSystem(c.SDL_INIT_AUDIO);
            gpa.destroy(st);
            return .{ .state = null };
        }

        for (&st.clips, 0..) |*clip, i| clip.* = Clip.load(clipPath(@enumFromInt(i)));
        for (&st.voices) |*v| {
            v.stream = c.SDL_CreateAudioStream(&mix_spec, &mix_spec);
            if (v.stream) |s| {
                if (!c.SDL_BindAudioStream(st.device, s)) {
                    c.SDL_DestroyAudioStream(s);
                    v.stream = null;
                }
            }
        }

        st.music_stream = c.SDL_CreateAudioStream(&mix_spec, &mix_spec);
        if (st.music_stream) |s| {
            _ = c.SDL_SetAudioStreamGetCallback(s, musicCallback, st);
            if (!c.SDL_BindAudioStream(st.device, s)) {
                c.SDL_DestroyAudioStream(s);
                st.music_stream = null;
            }
        }
        if (st.music_stream != null) st.music_thread = c.SDL_CreateThread(musicThread, "music-decode", st);
        return .{ .state = st };
    }

    pub fn deinit(self: *Audio) void {
        const st = self.state orelse return;
        // Join first: the decoder writes st.music. Destroying a stream waits
        // for any in-flight callback, so streams must die before the buffer.
        if (st.music_thread) |t| c.SDL_WaitThread(t, null);
        if (st.music_stream) |s| c.SDL_DestroyAudioStream(s);
        for (st.voices) |v| if (v.stream) |s| c.SDL_DestroyAudioStream(s);
        c.SDL_CloseAudioDevice(st.device);
        for (&st.clips) |*clip| clip.free();
        st.music.free();
        c.SDL_QuitSubSystem(c.SDL_INIT_AUDIO);
        st.gpa.destroy(st);
        self.state = null;
    }

    pub fn play(self: *Audio, sound: Sound) void {
        const st = self.state orelse return;
        const clip = &st.clips[@intFromEnum(sound)];
        if (clip.samples.len == 0) return;

        // Ticks start near zero; +1 keeps the zero-initialized "never played" slot distinct.
        const now = c.SDL_GetTicks() + 1;
        const last = &st.last_played_ms[@intFromEnum(sound)];
        if (now - last.* < repeat_window_ms) return;

        if (sound == .damage) {
            for (st.voices) |v| {
                if (v.sound == .damage and v.serial != 0 and isPlaying(v.stream)) return;
            }
        }

        var pick: ?*Voice = null;
        for (&st.voices) |*v| {
            if (v.stream == null) continue;
            if (!isPlaying(v.stream)) {
                pick = v;
                break;
            }
            if (pick == null or v.serial < pick.?.serial) pick = v;
        }
        const v = pick orelse return;

        _ = c.SDL_ClearAudioStream(v.stream);
        if (!c.SDL_PutAudioStreamData(v.stream, clip.samples.ptr, @intCast(clip.samples.len))) return;
        last.* = now;
        st.serial += 1;
        v.serial = st.serial;
        v.sound = sound;
    }

    pub fn setMusicMuted(self: *Audio, muted: bool) void {
        const st = self.state orelse return;
        if (st.music_stream) |s| _ = c.SDL_SetAudioStreamGain(s, if (muted) 0 else 1);
    }

    /// Decoded properties of a sound effect, or null if its file failed to load.
    pub fn soundInfo(self: *const Audio, sound: Sound) ?Info {
        const st = self.state orelse return null;
        const clip = st.clips[@intFromEnum(sound)];
        return if (clip.samples.len == 0) null else clip.info;
    }

    /// Music properties once the background decode has finished, else null.
    pub fn musicInfo(self: *const Audio) ?Info {
        const st = self.state orelse return null;
        return if (st.music_ready.load(.acquire)) st.music.info else null;
    }

    pub fn isActive(self: *const Audio) bool {
        return self.state != null;
    }
};

fn isPlaying(stream: ?*c.SDL_AudioStream) bool {
    return c.SDL_GetAudioStreamQueued(stream) > 0;
}

