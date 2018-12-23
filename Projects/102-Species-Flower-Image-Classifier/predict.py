import argparse
import numpy as np

from utils.transform_presets import T_RESIZE_CROP
from utils.image_processing import process_image
from utils.model_storage import load_checkpoint
from utils.model_building import reconstruct_model
from utils.model_inference import predict
from utils.file_util import load_json

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_image_dir", type=str, help="Path to input image", required=True)
parser.add_argument("--checkpoint_filepath", type=str, help="Path to checkpoint file", required=True)
parser.add_argument("--top_k", type=int, help="Number of top probabilities that should be displayed", default=5)
parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--category_names", type=str, help="Optional path to .json file with names for classes")

args = parser.parse_args()

if __name__ == "__main__":
	# Load data
	numpy_image = process_image(args.input_image_dir, T_RESIZE_CROP)

	# Load checkpoints
	checkpoint = load_checkpoint(args.checkpoint_filepath)

	# Restore model
	model = reconstruct_model(checkpoint)

	# Prediction
	probs, classes = predict(numpy_image, model, args.top_k, args.gpu)

	# Present results
	cat_to_id_map = None
	if args.category_names:
	    cat_to_id_map = load_json(args.category_names)

	print("\nResults for image '{}':".format(args.input_image_dir))
	prob_class_id_tuple_list = sorted([(p, c) for p, c in zip(probs, classes)], key=lambda t: t[0], reverse=True)
	for i, (probability, class_id) in enumerate(prob_class_id_tuple_list):
	    if cat_to_id_map is not None:
	        class_label = cat_to_id_map[str(class_id)] + " ({})".format(class_id)
	    else:
	        class_label = "(Class id: {})".format(class_id)
	    print("  {}. {} % - {}".format(i, np.round(probability * 100, 2), class_label))
