
py_file = file("python/src/main_estimate.py")
process estimate_density {
    label 'gpu'
    input:
        each path(FILE)
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
        tuple val(NAME), path("$NAME" + ".png"), path("$NAME" + ".npz"), path("$NAME" + ".pth"), path(FILE)

    script:
        NAME = "${FILE.baseName}__SCALE=${SCALE}__FOUR=${FOUR}__NORM=${NORM}__LR=${LR}__WD=${WD}__ACT=${ACT}__MAPPINGSIZE=${MAPPINGSIZE}_double" 
        """
        python $py_file \
            --csv0 $FILE \
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
    publishDir "result/double", mode: 'symlink'
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
    publishDir "result/double/${DATANAME[0]}/", mode: 'symlink'
    input:
        tuple val(NAMES), path(PNG), path(NPZ), path(WEIGHTS), path(FILE), val(DATANAME)

    output:
        tuple path("*.png"), path("*${DATANAME[0]}*_results.npz")

    script:
        """
        python $process double ${WEIGHTS[0]} ${WEIGHTS[1]} ${FILE[0]} ${FILE[1]} ${NPZ[0]} ${NPZ[1]}
        """
}


workflow double_f {
    take:
        data
        data
        scale
        fourier
        mapping_size
        norm
        lr
        wd
        act
        epoch
    main:
        estimate_density(data, scale, fourier, mapping_size, norm, lr, wd, act, epoch)
        //estimate_density.out[0]
        estimate_density.out[0].collectFile(name:"together.csv", keepHeader: true, skip:1).set{training_scores}
        selection(training_scores)
        selection.out[0] .splitCsv(skip:1, sep: ',')//.map{it -> it[0]} .view()
            .set{selected}
        estimate_density.out[1].join(selected, by: 0).groupTuple(by: 6).set{fused}
        post_processing(fused)
    emit:
        post_processing.output
}

    
data = Channel.fromPath("LyonN4/*.txt")
scale = [0.5] //0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
fourier = ["--fourier"]
norm = ["one_minus"]
lr = [0.1] //0.0001, 0.001, 0.01, 0.1, 1.0]
act = ["relu"]


workflow {
    main:
        double_f(data, scale, fourier, norm, lr, act)
}
