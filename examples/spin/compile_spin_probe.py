#!/usr/bin/env python3
"""Link a spin probe against the actual Makefiles or Ninja production objects."""
import argparse, pathlib, shlex, subprocess

def probe_command(command, build, source, output, root, command_cwd):
    tokens=shlex.split(command)
    if tokens[:2]==[':', '&&']:tokens=tokens[2:]
    if '&&' in tokens:tokens=tokens[:tokens.index('&&')]
    if not tokens or not any(t.endswith('principal.cc.o') for t in tokens):
        raise ValueError('not a recognized FeeBL production link command')
    result=[tokens[0]];removed=0;i=1
    while i<len(tokens):
        t=tokens[i]
        if t.endswith('principal.cc.o'):removed+=1;i+=1;continue
        if t=='-o':i+=2;continue
        if t.endswith(('.o','.a','.so')) and not t.startswith('-'):
            p=pathlib.Path(t);t=str(p if p.is_absolute() else (command_cwd/p).resolve())
        result.append(t);i+=1
    if removed!=1:raise ValueError('missing/duplicate production main object')
    return result+['-std=c++17','-O2','-fno-fast-math','-DUSE_NON_CONST','-I'+str(root/'src'),'-I'+str(build/'src'),str(source),'-o',str(output)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',type=pathlib.Path,required=True);ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--source',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.build=a.build.resolve()
    link=a.build/'src/CMakeFiles/feebl.dir/link.txt'
    if link.exists():cmd=link.read_text().strip();cwd=a.build/'src'
    elif (a.build/'build.ninja').exists():
        lines=subprocess.check_output(['ninja','-C',str(a.build),'-t','commands','src/feebl'],text=True).splitlines()
        links=[line for line in lines if 'principal.cc.o' in line and ' -o src/feebl ' in line]
        if len(links)!=1:raise ValueError('Ninja production link command is not unique')
        cmd=links[0];cwd=a.build
    else:raise ValueError('unsupported CMake generator or missing link receipt')
    args=probe_command(cmd,a.build,a.source.resolve(),a.output.resolve(),a.root.resolve(),cwd)
    # Static libraries must follow the probe source to resolve symbols used by it.
    src=args.pop(-3);args.insert(1,src)
    subprocess.run(args,cwd=cwd,check=True,timeout=120)
if __name__=='__main__':main()
