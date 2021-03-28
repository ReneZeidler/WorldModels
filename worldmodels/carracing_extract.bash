for i in $(seq 1 "${1:-64}");
do
  echo Starting worker "$i"
  CUDA_VISIBLE_DEVICES=-1 xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 extract.py -c configs/carracing.config &
  sleep 1.0
done
