

py_file = file("python/src/main_estimate.py")

process estimate_double_density_in_one {
    label 'gpu'
    input:
        tuple val(DATANAME), file(FILE0), file(FILE1)
        each SCALE
        each FOUR
        each MAPPINGSIZE
        each NORM
        each LR
        each WD
        each ACT
        each EPOCH

    output:
        path("$NAME" + ".csv")
        tuple val(NAME), val(DATANAME), path("$NAME" + ".npz"), path("$NAME" + ".pth"), path(FILE0), path(FILE1)
        tuple path("$NAME" + "0.png"), path("$NAME" + "1.png")

    script:
        CHUNK_ID = FILE0.baseName.split("-")[0]
        NAME = "${CHUNK_ID}-${DATANAME}__SCALE=${SCALE}__FOUR=${FOUR}__NORM=${NORM}__LR=${LR}__WD=${WD}__ACT=${ACT}__MAPPINGSIZE=${MAPPINGSIZE}_single"
        """
        python $py_file \
            --csv0 $FILE0 \
            --csv1 $FILE1 \
            --epochs $EPOCH \
            --scale $SCALE \
            --mapping_size $MAPPINGSIZE \
            $FOUR \
            --normalize $NORM \
            --lr $LR \
            --wd $WD \
            --activation $ACT\
            --name $NAME\
            --workers 8
        """
}


pyselect = file("python/src/selectbest.py")
process selection {
    publishDir "result/single/best", mode: 'symlink'
    input:
        path(CSV)

    output:
        path("selected.csv")
        path(CSV)
    script:
        """
        python $pyselect $CSV
        """
}

process = file("python/src/process_diff.py")


process post_processing {
    label 'gpu'
    publishDir "result/single/${DATANAMES[0]}/", mode: 'symlink'
    input:
        tuple val(NAMES), val(DATANAMES), path(NPZ), path(WEIGHTS), path(FILE0), path(FILE1), val(CHUNK_ID)

    output:
        tuple val(DATANAMES), path("*${DATANAMES}*_results.npz")
        path("*.png")
    script:
        """
        python $process single ${WEIGHTS} ${FILE0} ${FILE1} ${NPZ}
        """
}


process aggregate {
    publishDir "result/${TYPE}/", mode: 'symlink'
    input:
        tuple val(DATANAME), path(NPZ)
        val(TYPE)
    output:
        path("${DATANAME}_${TYPE}.csv")

    script:
        py_file = file("python/src/aggregate.py")
        """
        python $py_file $DATANAME $TYPE
        """
}

data = Channel.fromFilePairs("LyonN4/*{0,1}.txt")
scale = [0.5] //0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
fourier = ["--fourier"]
norm = ["one_minus"]
lr = [0.1] //0.0001, 0.001, 0.01, 0.1, 1.0]
act = ["relu"]


workflow single_f {
    take:
        paired_data
        scale
        fourier
        mapping_size
        norm
        lr
        wd
        act
        epoch
    main:
        estimate_double_density_in_one(paired_data, scale, fourier, mapping_size, norm, lr, wd, act, epoch)
        //estimate_density.out[0]
        estimate_double_density_in_one.out[0].collectFile(name:"together.csv", keepHeader: true, skip:1).set{training_scores}
        selection(training_scores)
        selection.out[0] .splitCsv(skip:1, sep: ',')
            .set{selected}
        estimate_double_density_in_one.out[1].join(selected, by: 0).set{fused}
        post_processing(fused)
        aggregate(post_processing.out[0].groupTuple(by: 0), "single")
    emit:
        aggregate.output
}

workflow {
    single_f(data, scale, fourier, norm, lr, act)
}
