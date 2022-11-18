
datapairs = Channel.fromFilePairs("data/clippeddata/*{0,1}.txt")

include { aggregate } from './pipeline_single_density.nf'

py_file = file("python/ot/ot_v2.py")

process optimal_transport {
    publishDir "result/OT/"
    label 'OT'
    input:
        tuple val(DATANAME), file(FILE0), file(FILE1)

    output:
        tuple val(DATANAME), path("$NAME" + ".npz")

    script:
        NAME = "${FILE0}__OT" 
        """
        python $py_file \
            --csv0 $FILE0 \
            --csv1 $FILE1 \
            --output $NAME
        """
}

workflow OT {
    take:
        paired_data
    main:
        optimal_transport(paired_data)
        aggregate(optimal_transport.out[0].groupTuple(by: 0), "OT")
    emit:
        aggregate.out[0]


}

workflow {
    main:
        OT(datapairs)
}
