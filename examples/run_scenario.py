#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

import logging
import pandas as pd

import converge_soe as csoe

import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--scaling', type=float, default='1.0')
parser.add_argument('indir', type=str)
parser.add_argument('outdir', type=str)

args = parser.parse_args()
scaling = args.scaling
indir = Path(args.indir)
outdir = Path(args.outdir)

outdir.mkdir(exist_ok=True, parents=True)

with (indir / 'network.json').open() as f:
    netw = json.load(f)

forecast = pd.read_csv(
    indir / 'forecast.csv', dtype={'load_id': str}
).set_index('load_id')
forecast *= scaling

def convert(df):
    for col in 'injection', 'consumption':
        df[col] = df[col].apply(lambda x: json.loads(x))
    return df
offers = convert(pd.read_csv(
    indir / 'offers.csv', dtype={'load_id': str}
)).set_index('load_id')

csoe.logger.addHandler(logging.FileHandler(outdir / 'log.txt'))

solver = csoe.SoeSolver(
    netw, forecast, offers, envelope_abs_max=50.0, solver_options={'linear_solver': 'ma27'}
)
status, results = solver.solve()

#solver.dump_opt_model(outdir / 'model.txt')
results.soe.to_csv(outdir/ 'soe.csv')
results.viol.to_csv(outdir/ 'viol.csv')
results.bus.to_csv(outdir/ 'bus.csv')
results.branch.to_csv(outdir/ 'branch.csv')
