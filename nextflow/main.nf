
include { double_f } from './pipeline_double_density.nf'
include { single_f } from './pipeline_single_density.nf'
include { optimal_transport } from './pipeline_single_density.nf'

data_paired = Channel.fromFilePairs("LyonN4/*{0,1}.txt")
data = Channel.fromPath("LyonN4/*.txt")
scale = [0.1, 0.5, 1.0, 5.0, 10.0]
fourier = ["--fourier"]
norm = ["one_minus"]
lr = [0.001, 0.01, 0.1, 1.0]
mapping_size = [128, 256, 512]
act = ["relu"]
epoch = [100]
wd = [0.0001]
params.extension = "ply"
ext = params.extension

paired_ply = Channel.fromFilePairs("data/full_data/pointCloud{0,1}.ply")
MAX_POINT = 500000

paired_txt = Channel.fromFilePairs("data/clippeddata/clippedMarco{0,1}.txt")

process from_ply_to_txt {
    input:
        tuple val(key), file(paired_file)
    output:
        path("*{0,1}.txt")
    script:
        pyfile = file("python/src/opening_ply.py")
        base0 = paired_file[0].baseName
        base1 = paired_file[1].baseName
        """
        python $pyfile ${base0}.ply ${base1}.ply
        """
}

process into_chunks {
    publishDir "result/chunks/"
    input:
        file(paired_file)
        val max_point
    output:
        path("*Chunks*{0,1}.txt")
    script:
        pyfile = file("python/src/split_grid.py")
        """
        python $pyfile ${paired_file[0]} ${paired_file[1]} ${max_point}

        """
}

process append_columns_headers {
    input:
        tuple val(key), file(paired_file)
    output:
        file("tmp*{0,1}.txt")

    script:
        base0 = paired_file[0].baseName
        base1 = paired_file[1].baseName
        """
        echo -e "X,Y,Z,R,G,B,label_ch\n\$(cat ${base0}.txt)" > tmp${base0}.txt
        echo -e "X,Y,Z,R,G,B,label_ch\n\$(cat ${base1}.txt)" > tmp${base1}.txt
        """
}


process final_table {
    publishDir "result/"
    input:
        file(results)
    output:
        file("final_results.csv")

    script:
        py_file = file("python/src/regroup.py")
        """
        python $py_file
        """
}


workflow {
    main:
        if (ext == "ply"){
            from_ply_to_txt(paired_ply)
            into_chunks(from_ply_to_txt.out, MAX_POINT)
            into_chunks.out.flatten().set{pointClouds}
            pointClouds.buffer( size: 2 , skip:0 ).set{pairedPointsclouds}
        } else {
            append_columns_headers(paired_txt)
            append_columns_headers.out.set{pairedPointsclouds}
            pairedPointsclouds.flatten().set{pointClouds}
        }
        double_f(pointClouds, scale, fourier, mapping_size, norm, lr, wd, act, epoch)
        single_f(pairedPointsclouds, scale, fourier, mapping_size, norm, lr, wd, act, epoch)
        optimal_transport(pairedPointsclouds)

        double_f.out.map{it -> it[2]}.concat(single_f.out.map{it -> it[2]}, optimal_transport.out).collect().set{results}
        final_table(results)
}
