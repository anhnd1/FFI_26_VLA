# Kaggle GPU job: set up openpi (unmodified, pinned) as a library + the LIBERO simulator, then run our code.
# Generated run.py (kaggle/push.py) unpacks the repo to $VLALAB, writes one config per arm to $ARMS_DIR/arm<i>.json and
# sets: MODE (eval | replay), N_ARMS, OPENPI_REPO, OPENPI_SHA.
#
# Arm i runs on GPU i % n_gpus; arms on the same GPU run one after the other. Outputs: /kaggle/working/<arm name>/.
#   eval:   policy server (vlalab.policy.server) -> latency benchmark (vlalab.sim.bench) -> LIBERO rollouts
#           (vlalab.sim.evaluate). Output format: src/vlalab/sim/recording.py.
#   replay: recorded calls from the attached dataset through the policy (vlalab.policy.replay), no simulator.
#
# Setup follows openpi's own instructions (README.md; examples/libero/{README.md,Dockerfile}), adapted to the Kaggle
# image (Ubuntu 24.04, Python 3.13), see the notes inline.
set -euxo pipefail
OUT=/kaggle/working
unset UV_SYSTEM_PYTHON   # preset by the Kaggle image; makes `uv pip` ignore the target venv
# openpi downloads gs://openpi-assets with gsutil; Kaggle's gsutil lacks compiled crcmod and then refuses composite
# objects. This is the boto option its error message points to.
printf '[GSUtil]\ncheck_hashes = if_fast_else_skip\n' > /tmp/boto.cfg
export BOTO_CONFIG=/tmp/boto.cfg

nvidia-smi | tee $OUT/gpu.txt
N_GPUS=$(nvidia-smi -L | wc -l)

if [ "$MODE" = eval ]; then
    # Simulator system packages (examples/libero/Dockerfile). On Ubuntu 24.04 libgl1-mesa-glx -> libgl1 +
    # libglx-mesa0 and libgles2-mesa-dev -> libgles-dev.
    apt-get -qq update
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y make g++ clang libosmesa6-dev libgl1 libglx-mesa0 libegl1 \
        libglew-dev libglfw3-dev libgles-dev libglib2.0-0 libsm6 libxrender1 libxext6 > /dev/null
    mkdir -p /usr/share/glvnd/egl_vendor.d
    echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

# --- openpi as a library: pinned commit, never patched ---
git clone -q $OPENPI_REPO /tmp/openpi
git -C /tmp/openpi checkout -q $OPENPI_SHA
git -C /tmp/openpi submodule update --init --recursive
{ git -C /tmp/openpi rev-parse HEAD; git -C /tmp/openpi submodule status;
  echo "local changes to openpi: $(git -C /tmp/openpi status --porcelain | wc -l)"; } > $OUT/versions.txt

pip install -q uv
(cd /tmp/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync)   # policy env, Python 3.11 (openpi README)
SERVER_PY=/tmp/openpi/.venv/bin/python
uv pip freeze --python $SERVER_PY > $OUT/server_env.txt

if [ "$MODE" = eval ]; then
    # LIBERO env, Python 3.8 (examples/libero/README.md, "Without Docker"); pinned in envs/.
    uv venv --python 3.8 /tmp/client
    CLIENT_PY=/tmp/client/bin/python
    uv pip sync --python $CLIENT_PY $VLALAB/envs/libero-client.txt $VLALAB/envs/libero-sim.txt \
        --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
    uv pip install --python $CLIENT_PY -e /tmp/openpi/packages/openpi-client -e /tmp/openpi/third_party/libero
    uv pip freeze --python $CLIENT_PY > $OUT/client_env.txt
    # LIBERO asks for its paths with input() on first import unless this file exists.
    export LIBERO_CONFIG_PATH=/tmp/libero
    mkdir -p $LIBERO_CONFIG_PATH
    L=/tmp/openpi/third_party/libero/libero/libero
    printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
        $L $L $L $L $L > $LIBERO_CONFIG_PATH/config.yaml
fi

# Download the checkpoints and tokenizer once, before parallel arms race for the same cache.
PYTHONPATH=$VLALAB/src $SERVER_PY - $ARMS_DIR/arm*.json <<'EOF'
import sys
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import download
from vlalab.config import ArmConfig
for ckpt in sorted({ArmConfig.load(f).policy.checkpoint for f in sys.argv[1:]}):
    print("checkpoint:", download.maybe_download(ckpt), flush=True)
PaligemmaTokenizer()
EOF

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 5 > $OUT/vram.csv &
VRAM_PID=$!

arm_name() { $SERVER_PY -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" $1; }

eval_arm() {  # $1 config, $2 gpu, $3 output dir
    local port=$((8000 + $2))
    # Without preallocation, nvidia-smi shows the VRAM JAX really uses instead of its default 75% reservation.
    CUDA_VISIBLE_DEVICES=$2 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$VLALAB/src \
        $SERVER_PY -m vlalab.policy.server --config $1 --port $port > $3/server.log 2>&1 &
    local server=$!
    until curl -sf localhost:$port/healthz > /dev/null; do   # clients wait forever, so stop if the server died
        kill -0 $server || { tail -50 $3/server.log; return 1; }
        sleep 10
    done
    PYTHONPATH=$VLALAB/src $SERVER_PY -m vlalab.sim.bench --config $1 --port $port --out $3/latency.csv \
        > $3/bench.log 2>&1
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$2 PYTHONPATH=$VLALAB/src:/tmp/openpi/third_party/libero \
        $CLIENT_PY -m vlalab.sim.evaluate --config $1 --port $port --out $3 > $3/client.log 2>&1 \
        || echo "evaluate exited with $?"
    kill $server || true
}

replay_arm() {  # $1 config, $2 gpu, $3 output dir
    local root
    root=$(dirname "$(find /kaggle/input -name vlalab_pack.json | head -1)")
    CUDA_VISIBLE_DEVICES=$2 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$VLALAB/src \
        $SERVER_PY -m vlalab.policy.replay --config $1 --data-root $root --out $3/replay.npz > $3/replay.log 2>&1
}

lane() {  # all arms assigned to GPU $1, sequentially
    for ((i = $1; i < N_ARMS; i += N_GPUS)); do
        local cfg=$ARMS_DIR/arm$i.json
        local out=$OUT/$(arm_name $cfg)
        mkdir -p $out
        cp $cfg $out/config.json
        ${MODE}_arm $cfg $1 $out || echo "arm $i failed"
    done
}

for ((g = 0; g < N_GPUS && g < N_ARMS; g++)); do lane $g & LANES[$g]=$!; done
for pid in "${LANES[@]}"; do wait $pid; done
kill $VRAM_PID || true

for f in $OUT/*/summary.json; do [ -f "$f" ] && cat "$f"; done
for f in $OUT/*/replay.log; do [ -f "$f" ] && tail -1 "$f"; done
true
