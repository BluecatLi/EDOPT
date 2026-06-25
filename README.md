# EDOPT: Event-camera 6-DoF Dynamic Object Pose Tracking

Paper: https://ieeexplore.ieee.org/abstract/document/10611511

Datasets: https://zenodo.org/records/10829647

```
@inproceedings{glover2024edopt,
  title={EDOPT: Event-camera 6-DoF Dynamic Object Pose Tracking},
  author={Glover, Arren and Gava, Luna and Li, Zhichao and Bartolozzi, Chiara},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={18200--18206},
  year={2024},
  organization={IEEE}
}
```

Assumes a known object model

Three simultaneous computations:

* EROS
* model projection
* state estimation

### Build the docker using:

```
cd EDOPT
docker build -t edopt:latest .

## if you want to build another remote branch for debugging ...
docker build -t edopt:latest --build-arg GIT_BRANCH=your/specified/remote/branch .
```
### Make and enter the container using:

```
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb -v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --network host --gpus all --name edopt edopt:latest
```
or
```
docker compose up -d
docker exec -it edopt /bin/bash
```
### How to build EDOPT
Terminal 1 (on docker container)
```
mkdir -p /usr/local/src/EDOPT/code/build
cd /usr/local/src/EDOPT/code/build
cmake ..
make
```

### How to run EDOPT
Terminal 1 (on docker container)
```
yarpserver
```

Terminal 2 (on docker container)
```
## if you do not have yarp config
yarp config {YOUR YARP IPADDRESS} {PORT}
yarp namespace {NAMESPACE}
yarp detect --write
## Run atis-bridge-sdk to receive event stream
atis-bridge-sdk --s 50
```

Terminal 3 (on docker container)
```
cd /usr/local/src/EDOPT/code/build
./edopt
```

### For development from docker container
Solution for Git Authentication

Terminal 1 (on docker container)
```
gh auth login
```
- Then, select as follows
  - ? Where do you use GitHub? > GitHub.com
  - ? What is your preferred protocol for Git operations on this host? > HTTPS
  - ? How would you like to authenticate GitHub CLI? > Login with a web browser
  - ! First copy your one-time code: XXXX-XXX
  - Open this link [https://github.com/login/device](https://github.com/login/device)
  - In the web browser, input one-time code to login

  ## Running EDOPT on recorded data (yarpdataplayer)

### One-time setup
- Recorded data lives in a folder containing `data.log` + `info.log`
  (e.g. `code/build/data/`). The source port is `/atis4/AE:o`.
- EDOPT's input port is `/ekom/AE:i` (set by `setName("/ekom")`).

### Each run (4 terminals)

1. **Start the YARP server**
```bash
   yarpserver
```

2. **Start EDOPT**
```bash
   cd code/build
   ./edopt --file run_name            # add --cstep for continuous-step version
```
   - This opens the input port `/ekom/AE:i`.
   - `--file run_name` logs the pose trajectory to `run_name` (+ `run_name.mp4`).
   - Press **G** in the SCARF window to start tracking; **P** prints the
     current pose in config format; **space** resets to initial pose.

3. **Start yarpdataplayer and load data**
```bash
   yarpdataplayer
```
   - `File → Open Directory` → select the **parent** folder of `data/`
     (NOT the `data` folder itself, NOT `data.log`).
   - A row with port `/atis4/AE:o` should appear. Do **not** press play yet.

4. **Connect the ports**
```bash
   yarp connect /atis4/AE:o /ekom/AE:i fast_tcp
```
   Verify:
```bash
   yarp name list | grep -E "atis4|ekom"     # both ports must exist
   yarp connect list | grep ekom             # connection must be listed
```

5. **Press play** in yarpdataplayer, then **press G** in the EDOPT window.

### Notes / gotchas
- Order matters: load data → connect → play. Re-opening data in
  yarpdataplayer recreates `/atis4/AE:o`, which **breaks the connection** —
  reconnect (step 4) after any stop/reload.
- To auto-reconnect, make the connection persistent:
```bash
  yarp connect /atis4/AE:o /ekom/AE:i fast_tcp --persist
```
- The camera calibration in the `.ini` must match the recording
  (ATIS4 = 640x480). A wrong `w/h/fx/...` makes the projection misalign
  and tracking fail silently.
- If yarpdataplayer crashes on opening a directory:
```bash
  pkill -9 yarpdataplayer && yarp clean --timeout 0.5
```
  or copy `data.log`+`info.log` into a clean isolated folder and open that.

### Plotting the logged trajectory
```bash
python3 plot_edopt_poses.py run_baseline run_cstep --save compare.png
```

### CLI flags
- `--cstep` : continuous variable-magnitude update (vs fixed discrete step)
- `--cstep_trust <f>` : trust-region size, x base step (default 3.0; try 1.0–1.5)
- `--cstep_eps <f>` : static-state deadband, relative to score (default 0.02)
- `--predict` : motion prediction (tested, no benefit at high rate — leave off)
- `--parallel` : threaded projection path (sequential is default)