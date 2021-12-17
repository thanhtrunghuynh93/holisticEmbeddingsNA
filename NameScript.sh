
for d in 0.2
do
	PD=graph_data/douban
	TRAINRATIO=${d}
	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict

	python -u network_alignment.py \
    --source_dataset ${PD}/online/graphsage/ \
    --target_dataset ${PD}/offline/graphsage/ \
    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
	NAME \
	--train_dict ${TRAIN} \
	--log \
	--cuda 
done
