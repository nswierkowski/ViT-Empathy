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
	--experiment_name vitbase_cls_seed42 \
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
	  --device cpu \
	  --path_preds data/analysis/vitbase_cls_seed42/preds_dir
