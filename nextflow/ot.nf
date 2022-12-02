
datapairs = Channel.fromFilePairs("data/clippeddata/*{0,1}.txt")

include { aggregate } from './pipeline_single_density.nf'

py_file = file("python/ot/ot_v2.py")

process optimal_transport {
    publishDir "result/OT/"
    label 'OT'
    input:
        tuple val(DATANAME), file(FILE0), file(FILE1)
        val(METHOD)
        val(EPS)
        val(UNB_EPS) 

    output:
        tuple val(DATANAME), val("OT"), path("$NAME" + ".npz")

    script:
        NAME = "${FILE0}__OT" 
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
