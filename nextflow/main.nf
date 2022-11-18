
include { double_f } from './pipeline_double_density.nf'
include { single_f } from './pipeline_single_density.nf'
include { OT } from './ot.nf'

// Parameters
scale = [1.0, 5.0]
fourier = ["--fourier"]
norm = ["one_minus"]
lr = [0.001, 0.01, 0.1]
mapping_size = [512]
act = ["relu"]
epoch = [100]
wd = [0.0001]
lambda_t = [0.0, 0.01, 1.0, 10.]
params.extension = "ply"
ext = params.extension
MAX_POINT = 30000

// Data
paired_ply = Channel.fromFilePairs("data/LyonS/*{0,1}.ply")
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
        tuple val(DATANAME), path("*Chunks*{0,1}.txt")//.collate(2)//.buffer( size: 2 , skip:0 )
    script:
        pyfile = file("python/src/split_grid.py")
        file0 = paired_file[0]
        file1 = paired_file[1]
        DATANAME = "${file0}".replaceFirst(/0.txt/, "")
        """
        python $pyfile ${file0} ${file1} ${max_point}
        """
}

process append_columns_headers {
    input:
        tuple val(key), file(paired_file)
    output:
        tuple val(key), file("tmp*0.txt"), file("tmp*1.txt")

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
        file("benchmark.csv")
    script:
        py_file = file("python/src/regroup_csv.py")
        """
        python $py_file

        """
}


workflow {
    main:
        if (ext == "ply"){
            from_ply_to_txt(paired_ply)
            into_chunks(from_ply_to_txt.out, MAX_POINT)
            into_chunks.out.flatMap{it -> it[1].stream().map(el -> [it[0], el]).collect()}.set{pointClouds}
            pointClouds.buffer(size: 2).map{it -> [it[0][0], it[0][1], it[1][1]]}.set{pairedPointsclouds}
        } else {
            append_columns_headers(paired_txt)

            append_columns_headers.out.set{pairedPointsclouds}
            pairedPointsclouds.map{it -> [[it[0], it[1]], [it[0], it[2]]]}.flatten().buffer(size: 2).set{pointClouds}
        }
        double_f(pointClouds, scale, fourier, mapping_size, norm, lr, wd, act, epoch)
        single_f(pairedPointsclouds, scale, fourier, mapping_size, norm, lr, wd, lambda_t, act, epoch)
        OT(pairedPointsclouds)
        double_f.out.concat(single_f.out, OT.out).collect().set{results}

        final_table(results)
}
