for i in $(seq 1 "${1:-64}");
do
  echo Starting worker "$i"
  CUDA_VISIBLE_DEVICES=-1 python3 extract.py -c configs/doom.config &
  sleep 1.0
done
