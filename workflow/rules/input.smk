from glob import glob

rule merge_pod5:
    input:
        pod5_dir="data/{run_name}/pod5/"
    output:
        pod5="data/{run_name}/merged.pod5"
    threads:
        8
    resources:
        disk_mb=500000,
        runtime=add_slack(500),
        mem_mb=50000,
    conda:
        "../envs/pod5.yml"
    shell:
        """
        pod5 merge {input.pod5_dir} -o {output.pod5} -t {threads}
        """

rule basecalling:
    input:
        pod5=rules.merge_pod5.output.pod5
    output:
        sam="results/{run_name}/calls.bam" 
    threads:
        8
    params:
        dorado_bin=config["dorado_path"],
        dorado_model=config["dorado_model"],
    resources:
        disk_mb=50000,
        runtime=add_slack(500),
        mem_mb=50000,
        slurm="gpus=2, nodes=1",
    shell:
        """
        {params.dorado_bin} basecaller {params.dorado_model} {input.pod5} --no-trim --emit-moves > {output.sam}
        """

rule sam_to_fastq:
    input:
        sam=rules.basecalling.output.sam
    output:
        fastq="results/{run_name}/calls.fastq.gz" #  calls.fastq.gz
    threads: 8
    conda:
        "../envs/minimap2.yml"
    resources:
        disk_mb=50000,
        runtime=add_slack(500),
        mem_mb=50000,
    shell:
        """
        samtools fastq -T "mv,ts,pi,sp,ns" {input.sam} | gzip -1 > {output.fastq}
        """

rule count_identifications:
    input:
        fastq=rules.sam_to_fastq.output.fastq,
    output:
        ident_count="results/{run_name}/ident_count.txt"
    params:
        seq=config["grep_seq"],
    shell:
        """
        zcat {input.fastq} | grep -c {params.seq} > {output.ident_count} 2>/dev/null || echo 0 > {output.ident_count}
        """

rule extract_read_ids_grep:
    input:
        fastq=rules.sam_to_fastq.output.fastq,
    output:
        read_ids="results/{run_name}/read-ids-grepmethod.txt"
    params:
        seq=config["grep_seq"],
    shell:
        """
        zcat {input.fastq} | grep -B1 {params.seq} | grep "^@" | awk '{{print substr($1, 2)}}' > {output.read_ids}
        """

rule align_reads:
    input:
        fastq=rules.sam_to_fastq.output.fastq,
    output:
        bam="results/{run_name}/{ref_name}/output.bam",
        bai="results/{run_name}/{ref_name}/output.bam.bai"
    threads: 16
    params:
        ref_fasta=lambda wildcards: config["ref_fasta"][wildcards.ref_name],
    conda:
        "../envs/minimap2.yml"
    resources:
        runtime=add_slack(500),
        mem_mb=50000,
    shell:
        """
        minimap2 -t {threads} -y -ax map-ont -k 9 -w 5 --secondary=no --sam-hit-only --MD -Y {params.ref_fasta} {input.fastq} | samtools sort -@ {threads} -o {output.bam}
        samtools index {output.bam}
        """

rule extract_read_ids_align:
    input:
        sam=rules.align_reads.output.bam,
    output:
        read_ids="results/{run_name}/{ref_name}/read-ids-alignmethod.txt"
    conda:
        "../envs/minimap2.yml"
    resources:
        disk_mb=10000,
        runtime=add_slack(1000),
        mem_mb=10000,
    shell:
        """
        samtools view -b -F 4 {input.sam} \
        | samtools fastq -n \
        | awk 'NR % 4 == 1 {{sub(/^@/, ""); print}}' \
        > {output.read_ids} || [ $? -eq 141 ]
        """

rule plot_grep:
    input:
        pod5=rules.merge_pod5.output.pod5,
        read_ids=rules.extract_read_ids_grep.output.read_ids,
    output:
        directory("results/{run_name}/grep-plots/")
    threads: 16
    params:
        script="workflow/scripts/peptide_signal_pipeline.py"
    resources:
        disk_mb=500000,
        runtime=add_slack(1000),
        mem_mb=100000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        """
        python {params.script} find-region {input.pod5} {output} -r {input.read_ids} 
        """

rule plot_align:
    input:
        pod5=rules.merge_pod5.output.pod5,
        read_ids=rules.extract_read_ids_align.output.read_ids,
        sam=rules.align_reads.output.bam,
    output:
        align_plots=directory("results/{run_name}/{ref_name}/align-plots/"),
        peptide_signals="results/{run_name}/{ref_name}/align-plots/peptide_signals.tsv"
    threads: 64
    params:
        script="workflow/scripts/peptide_signal_pipeline.py"
    resources:
        runtime=add_slack(2000),
        mem_mb=75000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        """
        python {params.script} find-region {input.pod5} {output.align_plots} -r {input.read_ids} --sam {input.sam} --signal_type pa
        """

rule plot_all:
    input:
        pod5=rules.merge_pod5.output.pod5
    output:
        directory("results/{run_name}/all-plots/")
    threads: 64
    params:
        script="workflow/scripts/peptide_signal_pipeline.py"
    resources:
        disk_mb=500000,
        runtime=add_slack(1000),
        mem_mb=100000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        """
        python {params.script} find-region {input.pod5} {output}
        """

rule aggregate_metadata:
    input:
        metadata="config/metadata_new.csv",
        align_plots=expand(
            "results/{run_name}/{ref_name}/align-plots/",
            run_name=[r for r, _ in CLASSIFY_RUN_REF_COMBOS],
            ref_name=[f for _, f in CLASSIFY_RUN_REF_COMBOS],
        )
    output:
        manifest="results/classification/metadata.csv"
    shell:
        """
        python workflow/scripts/update_metadata.py \
         {input.metadata} {output.manifest} \
         {input.align_plots}
        """

rule classify_signals_features_minirocket:
    """
    Classify peptide signals using featureLGBM and MiniRocket models.
    No GPU required - uses CPU-based feature extraction and lightweight boosting.
    """
    input:
        metadata="config/template.csv",
        align_plots=expand(
            "results/{run_name}/{ref_name}/align-plots/",
            run_name=[r for r, _ in CLASSIFY_RUN_REF_COMBOS],
            ref_name=[f for _, f in CLASSIFY_RUN_REF_COMBOS],
        )
    output:
        directory("results/classification/features_minirocket/")
    params:
        script="workflow/scripts/peptide_signal_pipeline.py",
    threads: 64
    resources:
        disk_mb=50000,
        runtime=add_slack(720),
        mem_mb=150000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        """
        python {params.script} classify-signals {input.metadata} \
        --output {output} --models featuresLGBM,MiniRocket --testing -1
        """

rule classify_signals_inceptiontime:
    """
    Classify peptide signals using InceptionTime model.
    Requires 3 GPUs for efficient deep learning-based signal classification.
    """
    input:
        metadata="config/template.csv",
        align_plots=expand(
            "results/{run_name}/{ref_name}/align-plots/",
            run_name=[r for r, _ in CLASSIFY_RUN_REF_COMBOS],
            ref_name=[f for _, f in CLASSIFY_RUN_REF_COMBOS],
        )
    output:
        directory("results/classification/inceptiontime/")
    params:
        script="workflow/scripts/peptide_signal_pipeline.py",
    threads: 128
    resources:
        disk_mb=50000,
        runtime=add_slack(1500),
        mem_mb=300000,
        slurm="gpus=8, nodes=1",
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        """
        python {params.script} classify-signals {input.metadata} \
        --output {output} --models InceptionTime --testing -1
        """

rule classify_signals_features_with_dtw:
    """
    Classify peptide signals using featureLGBM with DTW visualization.
    No GPU required - generates additional Dynamic Time Warping plots for interpretation.
    """
    input:
        metadata="config/template-dtw.csv",
        align_plots=expand(
            "results/{run_name}/{ref_name}/align-plots/",
            run_name=[r for r, _ in CLASSIFY_RUN_REF_COMBOS],
            ref_name=[f for _, f in CLASSIFY_RUN_REF_COMBOS],
        )
    output:
        directory("results/classification/features_dtw/")
    params:
        script="workflow/scripts/peptide_signal_pipeline.py",
    threads: 64
    resources:
        disk_mb=100000,
        runtime=add_slack(720),
        mem_mb=150000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        """
        python {params.script} classify-signals {input.metadata} \
        --output {output} --models featuresLGBM --plot_dtw --testing -1
        """

rule summary_csv:
    input:
        fastqs = expand("results/{run_name}/calls.fastq.gz", run_name=DATA_DIRS),
        sams   = expand("results/{run_name}/{ref_name}/output.bam",
                        run_name=DATA_DIRS, ref_name=ALL_REFS),
        idents = expand("results/{run_name}/ident_count.txt", run_name=DATA_DIRS),
        classification_summary="results/classification/per_peptide_test_metrics.csv",
    threads: 16
    resources:
        runtime=add_slack(100),
        mem_mb=50000,
    output:
        "results/summary_alignment.csv"
    conda:
        "../envs/pod5.yml"
    shell:
        "python workflow/scripts/make_summary.py"


rule dataset_basecalls_aligned_plr:
    input:
        fastqs=expand("results/{run_name}/calls.fastq.gz", run_name=DATA_DIRS),
        filtering_summaries=expand(
            "results/{run_name}/template/align-plots/filtering_summary.tsv",
            run_name=DATA_DIRS,
        )
    output:
        tsv="results/dataset_basecalls_aligned_PLR.tsv"
    threads: 16
    resources:
        runtime=add_slack(60),
        mem_mb=5000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        "python workflow/scripts/dataset_basecalls_aligned_plr.py --fastqs {input.fastqs} --summaries {input.filtering_summaries} --out {output.tsv} --workers {threads}"


rule count_alignment_thresholds:
    input:
        bam="results/{run_name}/{ref_name}/output.bam",
        bai="results/{run_name}/{ref_name}/output.bam.bai"
    output:
        tsv="results/{run_name}/{ref_name}/alignment_reach.tsv"
    params:
        ref=lambda wc: wc.ref_name,
        positions=[59, 70, 90, 100, 110, 120, 130]
    threads: 1
    resources:
        runtime=add_slack(20),
        mem_mb=1000,
    conda:
        "../envs/minimap2.yml"
    shell:
        """
        # Convert BAM to BED (gives chrom, start (0-based), end (1-based))
        bedtools bamtobed -i {input.bam} \
            | awk -v ref={params.ref} '$1 == ref && $6 == "+"' \
            > {output.tsv}.tmp.bed

        echo -e "threshold\tcount" > {output.tsv}
        for pos in {params.positions}; do
            # Count reads where end >= pos
            count=$(awk -v p="$pos" '$3 >= p' {output.tsv}.tmp.bed | wc -l)
            echo -e "${{pos}}\t${{count}}" >> {output.tsv}
        done
        """

rule aggregate_alignment_thresholds:
    input:
        expand(
            "results/{run_name}/{ref_name}/alignment_reach.tsv",
            run_name=DATA_DIRS,
            ref_name=ALL_REFS,
        )
    output:
        csv="results/alignment_reach_all.csv"
    shell:
        r"""
        echo "run,reference,threshold,count" > {output.csv}
        for f in {input}; do
            run=$(echo $f | cut -d/ -f2)
            ref=$(echo $f | cut -d/ -f3)
            tail -n +2 $f | awk -v r=$run -v ref=$ref \
                '{{print r","ref","$1","$2}}' >> {output.csv}
        done
        """


rule export_alignment_selected_counts:
    input:
        csv=rules.aggregate_alignment_thresholds.output.csv
    output:
        csv="results/alignment_selected_counts.csv"
    resources:
        runtime=add_slack(20),
        mem_mb=2000,
    conda:
        "../envs/pod5.yml"
    shell:
        "python workflow/scripts/export_alignment_selected_counts.py --input {input.csv} --out {output.csv} --template-ref template --template-threshold 59 --threading-ref template_N0_threading --threading-threshold 110 --revcomp-ref template_N0_revComTemplate --revcomp-threshold 110"

rule plot_alignment:
    input:
        "results/alignment_reach_all.csv"
    output:
        directory("results/alignment_plots/")
    threads: 4
    resources:
        runtime=add_slack(60),
        mem_mb=20000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    params:
        script="workflow/scripts/alignment_stats_summary.py"
    shell:
        "python {params.script} --input {input} --outdir {output}"


rule compute_extended_summary_row:
    """
    Per (run,ref) metric extraction. Produces results/{run}/{ref}/extended_metrics.json
    """
    input:
        fastq="results/{run_name}/calls.fastq.gz",
        pod5="data/{run_name}/merged.pod5",
        bam="results/{run_name}/{ref_name}/output.bam",
        peptide_signals="results/{run_name}/{ref_name}/align-plots/peptide_signals.tsv",
        seq_summary=lambda wc: glob(
            f"data/{wc.run_name}/sequencing_summary_*.txt"
        )
    output:
        json="results/{run_name}/{ref_name}/extended_metrics.json"
    threads: 1
    resources:
        runtime=add_slack(60),
        mem_mb=20000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        """
        python workflow/scripts/compute_extended_metrics.py \
          --fastq {input.fastq} \
          --bam {input.bam} \
          --pod5 {input.pod5} \
          --peptides {input.peptide_signals} \
          --seq-summary {input.seq_summary} \
          --out {output.json} || true
        """

rule aggregate_extended_summary:
    input:
        expand("results/{run_name}/{ref_name}/extended_metrics.json",
               run_name=DATA_DIRS, ref_name=ALL_REFS)
    output:
        csv="results/extended_summary.csv"
    resources:
        runtime=add_slack(60),
        mem_mb=20000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    shell:
        "python workflow/scripts/merge_extended_metrics.py --input {input} --out {output.csv}"


rule plot_extended_summary:
    input:
        summary="results/extended_summary.csv"
    output:
        directory("results/extended_plots/")
    resources:
        runtime=add_slack(60),
        mem_mb=20000,
    conda:
        "../envs/peptide-find-and-classify.yml"
    params:
        script="workflow/scripts/alignment_extended_summary.py"
    shell:
        "python {params.script} --input {input.summary} --outdir {output}"
