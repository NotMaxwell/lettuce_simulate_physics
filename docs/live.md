# Live view

Add `live=<port>` to any run, then open the page in a browser:

```bash
build/clang-release/lettuce configs/collision.ini n=2000000 method=tree steps=0 live=8080
# -> # live view at http://127.0.0.1:8080/
```

The page shows the simulation as it runs, with the step, time, energy error, virial ratio and
ms/step above it.

| Input | Action |
|---|---|
| drag | rotate |
| scroll | zoom |
| `space` / Pause | pause and resume the simulation |
| `s` / Step | advance one step (pauses) |
| `f` / Fit | refit the view to the bodies |
| `c` / Colour | density, or speed (blue slow, red fast) |
| Stop | end the run; outputs so far are kept and a final diagnostic is printed |

`steps=0` runs until Stop. A run with a step count keeps serving its final state when it ends
(`live_hold=false` exits instead). `live_paused=true` starts paused, so you can set up the camera first.

## Remote machines

The server listens on localhost only by default and has no authentication, so reach it through
SSH rather than opening the port to the network.

**Brev, or any machine you can ssh to:** forward the port from your laptop, then open
`http://localhost:8080/` locally.

```bash
ssh -L 8080:localhost:8080 <brev-host>
```

**ASA-X compute node:** the job runs on a compute node behind the login node.
1. Start an interactive job on the H100 partition, e.g. `srun -p <partition> --gres=gpu:h100:1 --pty bash`,
   and run there with `live=8080`. Note the node name (`hostname`).
2. From your laptop, jump through the login node:

   ```bash
   ssh -J <user>@<asax-login> -L 8080:localhost:8080 <user>@<compute-node>
   ```

   That works if the cluster lets you ssh to a node where you have a job, which many Slurm sites do.
3. If it doesn't, run with `live_bind=0.0.0.0`, which makes the node listen on its cluster-internal
   address, and forward through the login node instead:

   ```bash
   ssh -L 8080:<compute-node>:8080 <user>@<asax-login>
   ```

   Anyone else on the cluster network can then reach the page, and the page can stop your run.
   Check the site's policy before doing this.

## Cost

The simulation never waits for the browser. When the renderer is idle and the frame interval
(`live_fps`) has passed, the run copies the bodies off the GPU (32 bytes per body), and a separate
thread draws them on the CPU and JPEG-encodes the frame. The stream always sends the newest frame,
so a slow connection drops frames rather than falling behind. Rotating and zooming re-render the
last state, so the camera stays responsive even when steps are slow, and while paused.

Measured on a 4-core 2.1 GHz Xeon (CPU only):

| bodies | frame size | draw + encode |
|---|---|---|
| 20k | 1280x726 | 11–21 ms |
| 1M | 960x720 | 39 ms |
| 1M | 1920x1080 | 56 ms |

Drawing scales linearly with the body count. Past roughly 10–20M bodies each frame costs a
noticeable copy and a slow CPU draw; drawing on the GPU and copying only the finished image is the
fix, and isn't built yet.
