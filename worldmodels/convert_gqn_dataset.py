from utils import PARSER, get_path
from gqn.data_reader import to_tf_record
from data_reader import ensure_validation_split

args = PARSER.parse_args()

basedir = get_path(args, "record")

ensure_validation_split(basedir)

print("Converting training data...")
to_tf_record(basedir, args.gqn_context_size, consecutive_frames=args.gqn_consecutive_frames, mode='train')
print("Converting validation data...")
to_tf_record(basedir, args.gqn_context_size, consecutive_frames=args.gqn_consecutive_frames, mode='test')
