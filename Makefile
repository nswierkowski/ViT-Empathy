run_etl:
	PYTHONPATH=. python3 scripts/run_etl.py --image_dir data/faces --processed_dir data/processed

run_feature_extraction_trpakov:
	PYTHONPATH=. python scripts/run_feature_extraction.py \
	--image_dir data/faces \
	--processed_dir data/processed \
	--experiment_name vitface_cls_seed42 \
	--model_name trpakov/vit-face-expression \
	--pooling cls \
	--batch_size 64 \
	--num_workers 8 \
	--device cpu \
	--cache_root data/analysis/

run_feature_extraction_vitbase:
	PYTHONPATH=. python scripts/run_feature_extraction.py \
	--image_dir data/faces \
	--processed_dir data/processed \
	--experiment_name vitbase_cls_seed42_new_one \
	--model_name google/vit-base-patch16-224-in21k \
	--pooling cls \
	--batch_size 64 \
	--num_workers 8 \
	--device cpu \
	--cache_root data/analysis/

run_linear_probing_trpakov:
	PYTHONPATH=. python scripts/run_linear_probing.py \
	  --activations_dir data/analysis/vitface_cls_seed42 \
	  --device cpu \
	  --path_preds data/analysis/vitface_cls_seed42/preds_dir

run_linear_probing_vitbase:
	PYTHONPATH=. python scripts/run_linear_probing.py \
	  --activations_dir data/analysis/vitbase_cls_seed42 \
	  --device cuda \
	  --path_preds data/analysis/vitbase_cls_seed42/preds_dir

run_cka:
	PYTHONPATH=. python scripts/run_cka.py \
	  --pt1 data/analysis/vitbase_cls_seed42-fixed/train.pt \
	  --pt2 data/analysis/vitbase_cls_seed42-fixed/train.pt \
	  --experiment_name sad_vs_sad \
	  --compare_emotion_means happiness sadness \
	  --device cpu 

cka_runner:
	PYTHONPATH=. python scripts/run_cka.py \
	--pt1 data/analysis/vitbase_cls_seed42-fixed/train.pt \
	--pt2 data/analysis/vitbase_cls_seed42-fixed/train.pt \
	--experiment_name neutrality_m_vs_f \
	--compare_sex_means neutrality \
	--device cpu \

single_patching_analysis:
	PYTHONPATH=. python scripts/run_single_patching_exp.py \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/single_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--device cuda

single_emb_patching_analysis:
	PYTHONPATH=. python scripts/run_single_patching_exp.py \
	--embedding \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/emb_single_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--device cuda

group_patching_analysis:
	PYTHONPATH=. python scripts/run_group_patching_exp.py \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/group_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--json_path data/face_patches/face_parsing/face_patches_ALL.json \
	--device cuda

group_emb_patching_analysis:
	PYTHONPATH=. python scripts/run_group_patching_exp.py \
	--embedding \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/group_emb_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--json_path data/face_patches/face_parsing/face_patches_ALL.json \
	--device cuda

group_cls_patching_analysis:
	PYTHONPATH=. python scripts/run_group_patching_exp.py \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/group_cls_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--json_path data/face_patches/face_parsing/face_patches_ALL.json \
	--device cuda \
	--use_cls_token

group_emb_cls_patching_analysis:
	PYTHONPATH=. python scripts/run_group_patching_exp.py \
	--embedding \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/group_emb_cls_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--json_path data/face_patches/face_parsing/face_patches_ALL.json \
	--device cuda \
	--use_cls_token

whole_img_analysis:
	PYTHONPATH=. python scripts/run_group_patching_exp.py \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/whole_img_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--json_path data/face_patches/whole_image/whole_image_ALL.json \
	--device cuda

whole_img_cls_analysis:
	PYTHONPATH=. python scripts/run_group_patching_exp.py \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/whole_img_cls_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--json_path data/face_patches/whole_image/whole_image_ALL.json \
	--device cuda \
	--use_cls_token 

whole_img_embedding_analysis:
	PYTHONPATH=. python scripts/run_group_patching_exp.py \
	--embedding \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/whole_img_emb_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--json_path data/face_patches/whole_image/whole_image_ALL.json \
	--device cuda

cls_patching_analysis:
	PYTHONPATH=. python scripts/run_cls_patching_exp.py \
	--model_name google/vit-base-patch16-224-in21k \
	--path_to_save_results data/experiments/cls_patching/vitbase_neutrality \
	--original_image_path data/faces/006_m_f_h_b.jpg \
	--corrupted_image_path data/faces/006_m_f_s_b.jpg \
	--preds_path data/analysis/vitbase_cls_seed42/preds_dir/ \
	--start_layer 0 \
	--last_layer 11 \
	--device cuda

run_face_parser:
	PYTHONPATH=. python scripts/run_face_parser.py \
	--image_dir data/faces \
	--processed_dir data/processed \
	--path_to_save_json data/face_patches/face_parsing/face_patches_train.json \
	--ids_to_save 1 2 4,5 6,7 10,11,12 17 \
	--splits train \
	--save_mask

run_face_parser_whole_image:
	PYTHONPATH=. python scripts/run_face_parser.py \
	--image_dir data/faces \
	--processed_dir data/processed \
	--path_to_save_json data/face_patches/whole_image/whole_image_test.json \
	--ids_to_save 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 \
	--splits test \
