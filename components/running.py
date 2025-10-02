from process_real2real_pipeline import both_real_main
from process_real2synth_pipeline import syn_real_main


initial_selected_weights = 0.5
input_directory_path = "./components/dataset/specific_data" 
output_directory_path = "./components/dataset/data_manipulation"

syn_real_main(initial_selected_weights, input_directory_path, output_directory_path)


initial_selected_weights = 0.5
input_directory_path = "./components/dataset/specific_data" 
output_directory_path = "./components/dataset/data_manipulation"
number_of_videos = 3

both_real_main(initial_selected_weights, input_directory_path, output_directory_path, number_of_videos) 