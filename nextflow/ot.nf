
data = Channel.fromFilePairs("data/clippeddata/*{0,1}.txt")
data.map{it -> it[1]}.set {datapairs}

py_file = file("python/ot/ot_change_detection.py")

process optimal_transport {
    publishDir "result/OT/"
    label 'OT'
    input:
        tuple file(FILE0), file(FILE1)

    output:
        path("$NAME" + ".npz")

    script:
        NAME = "${FILE0.baseName}__OT" 
        """
        python $py_file \
            --csv0 $FILE0 \
            --csv1 $FILE1 \
            --output $NAME
        """
}



workflow {
    main:
        optimal_transport(datapairs)
}
