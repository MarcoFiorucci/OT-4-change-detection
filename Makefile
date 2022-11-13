
full_data:
	nextflow run nextflow/main.nf -resume --extension ply -profile raiden

clipped:
	nextflow run nextflow/main.nf -resume --extension txt -profile raiden