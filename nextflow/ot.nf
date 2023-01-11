
datapairs = Channel.fromFilePairs("data/clippeddata/*{0,1}.txt")

py_file = file("python/ot/ot_cd.py")

process optimal_transport {
    publishDir "result/OT/"
    label 'OT'
    input:
        tuple val(DATANAME), file(FILE0), file(FILE1)
        val(METHOD)
        each EPS
        each UNB_EPS 

    output:
        tuple val(DATANAME), val("OT__${EPS}__${UNB_EPS}"), path("$NAME" + ".npz")

    script:
        NAME = "${FILE0}__${EPS}__${UNB_EPS}__OT" 
        """
        python $py_file \
            --method $METHOD \
            --csv0 $FILE0 \
            --csv1 $FILE1 \
            --output $NAME \
            --epsilon $EPS \
            --epsilon_unbalanced $UNB_EPS 
        """
}

process aggregate {
    publishDir "result/individual/${METHOD}/", mode: 'symlink'
    input:
        tuple val(DATANAME), val(METHOD), path(NPZ)
    output:
        path("${DATANAME}_${METHOD}.csv")
        path("${DATANAME}_${METHOD}_chunkinfo.csv")

    script:
        py_file = file("python/src/aggregate.py")
        """
        python $py_file $DATANAME $METHOD
        """
}


workflow OT {
    take:
        paired_data
        method
        epsilon
        unb_epsilon
    main:
        optimal_transport(paired_data, method, epsilon, unb_epsilon)
        aggregate(optimal_transport.out[0].groupTuple(by: [0, 1]))

    emit:
        aggregate.out[0]


}

workflow {
    main:
        OT(datapairs)
}
