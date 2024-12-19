import shutil
import traceback
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import Namespace, ArgumentParser

from src.libs.parse_opt import get_list_sessions

from src.capture_images import CaptureImages
from src.savers import MultilabelPredictions
from src.multilabel_classifier import MultiLabelClassifierCUDA
from src.libs.predictions_raster_tools import create_rasters_for_classes

from src.multilabel_classifier import MultiLabelClassifierTRT
from src.libs.common_cuda import cuda_initialisation



def parse_args() -> Namespace:

    # Parse command line arguments.
    ap = ArgumentParser(description="Drone inference", epilog="Thanks to use it!")

    # Input.
    arg_input = ap.add_mutually_exclusive_group(required=True)
    arg_input.add_argument("-efol", "--enable_folder", action="store_true", help="Take all images from a folder of a seatizen session")
    arg_input.add_argument("-eses", "--enable_session", action="store_true", help="Take all images from a single seatizen session")
    arg_input.add_argument("-ecsv", "--enable_csv", action="store_true", help="Take all images from session in csv file")

    # Path of input.
    ap.add_argument("-pfol", "--path_folder", default="/media/bioeos/E/drone_serge_test/", help="Load all images from a folder of sessions")
    ap.add_argument("-pses", "--path_session", default="/media/bioeos/E/drone_serge_test/20231201_REU-HERMITAGE_UAV_01", help="Load all images from a single session")
    ap.add_argument("-pcsv", "--path_csv_file", default="./csv_inputs/stleu.csv", help="Load all images from session write in the provided csv file")


    # Choose how to used multilabel model.
    ap.add_argument("-mlu", "--multilabel_url", default="lombardata/drone-DinoVdeau-from-probs-large-2024_11_15-batch-size32_freeze_probs", help="Hugging face repository")
    ap.add_argument("-mlgpu", "--multilabel_gpu", action="store_true", help="Speedup inference with tensorrt")
    ap.add_argument("-nml", "--no_multilabel", action="store_true", help="Didn't used multilabel model")


    # Orthophoto arguments.
    ap.add_argument('-crs', '--matching_crs', type=str, default="32740", help="Default CRS of the project.")
    ap.add_argument('-tsm', '--tiles_size_meters', type=float, default=1.5, help='Tile size in meters')
    ap.add_argument('-hs', '--h_shift', type=float, default=0, help='Horizontal overlap. Default 0')
    ap.add_argument('-vs', '--v_shift', type=float, default=0, help='Vertical overlap.')
    ap.add_argument('-bptp', '--black_pixels_threshold_percentage', type=float, default=5, help="Don't keep tile if we have a bigger percentage of black pixels than threshold.")
    ap.add_argument('-wptp', '--white_pixels_threshold_percentage', type=float, default=5, help="Don't keep tile if we have a bigger percentage of white pixels than threshold.")


    # Optional arguments.
    ap.add_argument("-np", "--no-progress", action="store_true", help="Hide display progress")
    ap.add_argument("-ns", "--no-save", action="store_true", help="Don't save annotations")
    ap.add_argument("-npr", "--no_prediction_raster", action="store_true", help="Don't produce predictions rasters")
    ap.add_argument("-c", "--clean", action="store_true", help="Clean pdf preview and predictions files")
    ap.add_argument("-is", "--index_start", default="0", help="Choose from which index to start")
    ap.add_argument("-ip", "--index_position", default="-1", help="if != -1, take only session at selected index")
    ap.add_argument("-bs", "--batch_size", default="1", help="Numbers of frames processed in one time")

    return ap.parse_args()

def pipeline_seatizen(opt: Namespace):
    print("\n-- Parse input options", end="\n\n")
    
    batch_size = int(opt.batch_size) if opt.batch_size.isnumeric() else 1

    cuda_initialisation(opt.multilabel_gpu)

    print("\n-- Load the pipeline ...", end="\n\n")
    capture_images = CaptureImages(opt)

    # Load Hugging face model.
    multilabel_model = None 
    if opt.no_multilabel:
        multilabel_model = None 
    elif opt.multilabel_gpu:
        multilabel_model = MultiLabelClassifierTRT(opt.multilabel_url, batch_size)
    else:
        multilabel_model = MultiLabelClassifierCUDA(opt.multilabel_url, batch_size)

    multilabel_saver = None if opt.no_multilabel else MultilabelPredictions(multilabel_model.classes_name)
    if opt.no_save:
        multilabel_saver = None

    # Stat
    sessions_fail = []
    list_session = get_list_sessions(opt)
    index_start = int(opt.index_start) if opt.index_start.isnumeric() and int(opt.index_start) < len(list_session) else 0
    index_position = int(opt.index_position)-1 if opt.index_position.isnumeric() and \
                                            int(opt.index_position) > 0 and \
                                            int(opt.index_position) <= len(list_session) else -1
    sessions = list_session[index_start:] if index_position == -1 else [list_session[index_position]]
    print("\n-- Start inference !", end="\n\n")

    for session in sessions:

        print(f"\nLaunched session {session.name}\n\n")

        # Clean sessions if needed
        path_IA = Path(session, "PROCESSED_DATA/IA")
        if Path.exists(path_IA) and opt.clean:
            print("\t-- Clean session \n\n")
            shutil.rmtree(path_IA)
        path_IA.mkdir(exist_ok=True, parents=True)

        multilabel_scores_csv_name = Path(session, "PROCESSED_DATA/IA", f"{session.name}_{opt.multilabel_url.replace('/', '_')}_scores.csv")

        # Setup pipeline for current session
        capture_images.setup(session)
        if multilabel_saver:
            multilabel_saver.setup(multilabel_scores_csv_name) 

        pipeline = (
            capture_images |
            multilabel_model | 
            multilabel_saver
        )

        # Iterate through pipeline
        start_t = datetime.now()
        print("\t-- Start prediction session \n\n")
        progress = tqdm(total=capture_images.frame_count,
                        disable=opt.no_progress)
        
        try:
            for _ in pipeline:
                progress.update(1)
        except StopIteration:
            return
        except KeyboardInterrupt:
            return
        finally:
            progress.close()

        # Pipeline cleanup.
        if multilabel_saver:
            multilabel_saver.cleanup()

        print(f"\n -- Elapsed time: {datetime.now() - start_t} seconds\n\n")

        try:
          
            # Create raster predictions
            print("\t-- Creating raster for each class \n\n")
            if not opt.no_prediction_raster:
                pass
                create_rasters_for_classes(multilabel_scores_csv_name, multilabel_model.classes_name, path_IA, session.name, 'linear')
            
            print(f"\nSession {session.name} end succesfully ! ", end="\n\n\n")

        except Exception:
            print(traceback.format_exc(), end="\n\n")

            sessions_fail.append(session.name)
    
    # Stat
    print("\nEnd of process. On {} sessions, {} fails. ".format(len(sessions), len(sessions_fail)))
    if (len(sessions_fail)):
        [print("\t* " + session_name) for session_name in sessions_fail]
    
if __name__ == "__main__":
    args = parse_args()
    pipeline_seatizen(args)