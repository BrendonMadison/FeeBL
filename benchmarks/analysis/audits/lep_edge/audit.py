"""Check the LEP focusing reference against the primary PDF's vector histogram.

The externally supplied source figure is identified by its SHA256. No reference
data are changed, and every invocation requires a new output directory.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VECTOR_SHA256 = "0cc1263aade2f9574b5213f7b473d2f37eb65c4a129bbfc368268f28cd65769e"
COLOURS = {(0.,0.,0.):"all", (1.,0.,0.):"early", (0.,0.,1.):"late"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def recover_operations(operations):
    """Read the three staircase strokes in the known source's local coordinates.

    Frame and traces share its initial translation and quarter-turn. Their
    untransformed coordinates avoid roundoff from the older rotated export.
    This is a checked extractor for one source figure, not a general PDF parser.
    """
    colour,stack,lines,point,paths,frames=None,[],[],None,[],[]
    for args,operation in operations:
        if operation == b"q":
            stack.append(colour)
        elif operation == b"Q":
            require(bool(stack),"Unbalanced PDF graphics state")
            colour=stack.pop()
        elif operation == b"RG":
            colour=tuple(map(float,args))
        elif operation == b"re":
            rectangle=tuple(map(float,args))
            if rectangle[0]>70 and rectangle[2]>300:
                frames.append(rectangle)
        elif operation == b"m":
            point,lines=tuple(map(float,args)),[]
        elif operation == b"l":
            end=tuple(map(float,args)); lines.append((point,end)); point=end
        elif operation in (b"S",b"s"):
            if len(lines)>80:
                paths.append((colour,lines))
            lines=[]
        elif operation in (b"n",b"f",b"f*",b"b*"):
            lines=[]
    require(len(set(frames))==1,"Expected one unique plotting frame")
    require(len(paths)==3,"Expected three histogram strokes")
    x0,y0,width,height=frames[0]
    require(width>0 and height>0,"Invalid frame dimensions")
    recovered=[]
    for colour,segments in paths:
        require(colour in COLOURS,"Unexpected histogram stroke colour")
        for start,end in segments:
            if abs(start[1]-end[1])>.002 or abs(start[0]-end[0])<.2:
                continue
            low,high=sorted((start[0],end[0]))
            require(x0-.001<=low<high<=x0+width+.001,"Histogram segment outside the frame")
            low,high=-20+(low-x0)*50/width,-20+(high-x0)*50/width
            recovered.append(dict(series=COLOURS[colour],theta_lo_urad=low,
                theta_hi_urad=high,theta_centre_urad=(low+high)/2,
                ordinate_au=max(0.,(start[1]-y0)*1e5/height)))
    require(Counter(r["series"] for r in recovered)==dict(all=100,early=100,late=100),
            "Expected 100 bins per source series")
    return recovered,[x0,y0,width,height]


def normalized_density(rows):
    values=[float(r["ordinate_au"]) for r in rows]
    widths=[float(r["theta_hi_urad"])-float(r["theta_lo_urad"]) for r in rows]
    require(all(math.isfinite(v) and v>=0 for v in values),"Invalid histogram ordinates")
    require(all(math.isfinite(w) and w>0 for w in widths),"Invalid histogram widths")
    total=math.fsum(v*w for v,w in zip(values,widths))
    require(total>0,"Empty source histogram")
    return [v/total for v in values]


def run(vector_pdf,reference_csv,reduction_csv,output):
    require(digest(vector_pdf)==VECTOR_SHA256,"Source PDF hash differs from the audited primary figure")
    from pypdf import PdfReader
    reader=PdfReader(vector_pdf)
    require(len(reader.pages)==1,"Expected the standalone one-page source figure")
    recovered,frame=recover_operations(reader.pages[0].get_contents().operations)
    original=read_csv(reference_csv)
    require(len(original)==len(recovered),"Reference/source row count differs")
    differences=[]
    for a,b in zip(original,recovered):
        require(a["series"]==b["series"],"Reference/source series order differs")
        differences.append({k:abs(float(a[k])-b[k]) for k in b if k!="series"})
    maxima={k:max(row[k] for row in differences) for k in differences[0]}
    require(max(maxima[k] for k in maxima if k!="ordinate_au")<.0002,"CSV/source coordinate discrepancy")
    require(maxima["ordinate_au"]<.05,"CSV/source ordinate discrepancy")
    all_source=[r for r in recovered if r["series"]=="all"]
    all_csv=[r for r in original if r["series"]=="all"]
    density,old_density=normalized_density(all_source),normalized_density(all_csv)
    rows=read_csv(reduction_csv)
    require(len(rows)==100,"Expected the original 100-bin comparison")
    for a,b,ordinate in zip(all_csv,rows,old_density):
        require(float(a["theta_lo_urad"])==float(b["bin_low"])
                and float(a["theta_hi_urad"])==float(b["bin_high"]),"Display/reference bin mismatch")
        display_density=float(b["reference"])/(float(b["bin_high"])-float(b["bin_low"]))
        require(abs(ordinate-display_density)<1e-12,"Display/reference normalization mismatch")
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/"independent_vector_histogram.csv",recovered)
    selected=[dict(original_bin_low=float(a["theta_lo_urad"]),original_bin_high=float(a["theta_hi_urad"]),
        original_ordinate=float(a["ordinate_au"]),pdf_bin_low=b["theta_lo_urad"],
        pdf_bin_high=b["theta_hi_urad"],pdf_ordinate=b["ordinate_au"])
        for a,b in zip(all_csv,all_source) if 23<=float(a["theta_lo_urad"])<=27]
    write_csv(output/"edge_bins.csv",selected)
    receipt=dict(status="PASS_WITHIN_VECTOR_COORDINATE_TOLERANCES",
        primary_source="https://arxiv.org/abs/1908.01704v4",doi="10.1016/j.physletb.2019.135068",
        source_figure="Figure 2 left, article PDF page 7",
        original_vector_sha256=digest(vector_pdf),reference_csv_sha256=digest(reference_csv),
        reduction_csv_sha256=digest(reduction_csv),audit_script_sha256=digest(Path(__file__)),
        source_paths=3,bins_per_series=100,calibration=dict(frame_local_pdf_points=frame,
            x_range_urad=[-20,30],ordinate_range=[0,100000]),
        numerical_tolerances=dict(coordinate_urad=.0002,ordinate_au=.05,display_density=1e-12),
        max_absolute_csv_minus_pdf=maxima,
        max_absolute_normalized_density_difference=max(abs(a-b) for a,b in zip(density,old_density)),
        display_reference_matches_original_csv=True,replacement_recommended=False,
        largest_observed_pull_bin=max(rows,key=lambda r:abs(float(r["pull"]))),
        interpretation="The sharp published edge is present in the primary vector histogram. The endpoint excess in the simulation is not explained by this digitization or bin-mapping check; its physical or numerical cause remains separate.",
        output_sha256={name:digest(output/name) for name in ("independent_vector_histogram.csv","edge_bins.csv")})
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-pdf",type=Path,required=True,help="Externally obtained original vector figure; exact SHA256 required")
    parser.add_argument("--reference-csv",type=Path,default=ROOT/"benchmarks/reference/lep_final_focusing_distribution.csv")
    parser.add_argument("--reduction-csv",type=Path,default=ROOT/"benchmarks/example_output/qed_figure_campaign/reduction/lep_focusing_distribution_pull.csv")
    parser.add_argument("--output",type=Path,required=True,help="New output directory; never overwritten")
    args=parser.parse_args()
    result=run(args.vector_pdf,args.reference_csv,args.reduction_csv,args.output)
    print(json.dumps(dict(status=result["status"],receipt=str(args.output/"receipt.json"))))


if __name__=="__main__":
    main()
