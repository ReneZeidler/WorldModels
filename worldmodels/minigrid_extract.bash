for i in $(seq 1 "${1:-64}");
do
  echo Starting worker "$i simple"
  CUDA_VISIBLE_DEVICES=-1 python3 extract.py -c configs/minigrid_simple.config &
  echo Starting worker "$i rooms"
  CUDA_VISIBLE_DEVICES=-1 python3 extract.py -c configs/minigrid_rooms.config &
  sleep 1.0
done
