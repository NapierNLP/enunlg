#!/usr/bin/env bash
set -euo pipefail

# Definitions

declare -A SUPPORTED_DATASETS=()
for LABEL in cacapo cued e2e e2e-cleaned e2e-enriched neural-methodius webnlg2023 webnlg-enriched
do
  SUPPORTED_DATASETS["${LABEL}"]=1
done

function fetch_dataset {
  case $1 in
    cacapo)
      echo "CACAPO's full data requires manual download from: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/LIBYHP"
      echo "Please save the ZIP file from that page to datasets/raw/cacapo.zip"
      read -p "Did you save the data to the correct location with the correct filename? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1
      if [ -e "cacapo.zip" ]
      then
        unzip cacapo.zip
      else
        echo "You responded '${confirm}, indicating that you downloaded the file to the correct location and filename,"
        echo "however, we cannot find the file there. Please try again."
        exit 1
      fi ;;
    cued)
      if [ ! -d RNNLG ]
      then
        git clone https://github.com/shawnwun/RNNLG.git
      fi ;;
    e2e)
      wget https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip
      unzip e2e-dataset.zip ;;
    e2e-cleaned)
      git clone git@github.com:tuetschek/e2e-cleaning.git ;;
    e2e-enriched)
      if [ ! -d EnrichedE2E ]
      then
        git clone https://github.com/ThiagoCF05/EnrichedE2E.git
      fi;;
    neural-methodius)
      git clone git@github.com:aleksadre/methodiusNeuralINLG2021 ;;
    webnlg2023)
      git clone git@github.com:WebNLG/2023-Challenge.git ;;
    webnlg-enriched)
      if [ ! -d webnlg ]
      then
        git clone https://github.com/ThiagoCF05/webnlg.git
      fi;;
  esac
}


function process_dataset {
  case $1 in
    cued)
      mkdir -p datasets/glove-vectors
      tail -n10164 datasets/raw/RNNLG/vec/vectors-80.txt > datasets/glove-vectors/rnnlg.vectors-80.txt ;;
    e2e-enriched)
      cp -rf datasets/raw/EnrichedE2E datasets/processed/.
      # Fix errors  in corpus annotation
      # TODO check usage of __s__
      sed -i -e 's/__CUSTOMER___ RATING/__CUSTOMER_RATING__/g' -e 's/__CUSTOMER___ __RATING\([a-z\-]*\)__/__CUSTOMER_RATING__ \1/g' -e 's/ CUSTOMER_RATING //g' -e 's/\([ .]\)__ /\1/g' -e 's/__ @ FAMILYFRIENDLY /__ @ __FAMILYFRIENDLY__ /g' -e 's/<text>The Wrestlers is \(a decent\|an average\) \(family-\|child \)friendly \(place\|venue\).</<text>The Wrestlers is \1 @ \2friendly \3.</g' -e 's/Not family friendly Alimentum/Not family friendly @ Alimentum/g' -e 's/high-priced kids-friendly/high-priced @ kids-friendly/g' -e "s/Its a /It's a /g" -e "s/ __NAME__s / __NAME__ 's /g" -e "s/ __s__ / /g" -e 's/verage family friendly/verage @ family friendly/g' -e 's/ it high/ it @ high/g' -e 's/__..but/but/g' -e 's/\.\.\.but/, but/g' -e 's/for children it/for children @ it/g' -e 's/__CUSTOMER_RATING__and/__CUSTOMER_RATING__ and/g' -e 's/5and/5 and/g' -e "s/ ' \(s\|re\) / '\1 /g" -e 's/ __FAMILYFRIENDLY__t family - friendly / __FAMILYFRIENDLY__ /g' -e 's/ \(__PRICERANGE__ly\|__PRICERANGEly__\) / __PRICERANGE__ /g' datasets/processed/EnrichedE2E/*/*.xml
      head -n2405 datasets/processed/EnrichedE2E/dev/3attributes.xml > tmp.3atts
      echo '        <sentence>' >> tmp.3atts
      echo '          <input attribute="name" tag="__NAME__" value="The Phoenix"/>' >> tmp.3atts
      echo '          <input attribute="area" tag="__AREA__" value="riverside"/>' >> tmp.3atts
      echo '          <input attribute="customer rating" tag="__CUSTOMER_RATING__" value="5 out of 5"/>' >> tmp.3atts
      echo '        </sentence>' >> tmp.3atts
      tail -n2923 datasets/processed/EnrichedE2E/dev/3attributes.xml >> tmp.3atts
      mv tmp.3atts datasets/processed/EnrichedE2E/dev/3attributes.xml;;
    webnlg-enriched)
      cp -rf datasets/raw/webnlg datasets/processed/.
      # Fix errors in corpus annotation
      sed -i -e 's/"BRIDGE-"/"BRIDGE-2"/g' -e 's/ BRIDGE- / BRIDGE-2 /g' datasets/processed/webnlg/data/v1.6/en/dev/3triples/Building.xml
#      sed -i -e 's/\(AGENT\|BRIDGE\|PATIENT\)-\([0-9]\)/__\1_\2__/g' datasets/processed/webnlg/data/v1.6/en/*/*/*.xml;;
  esac
}


function usage {
    echo "Usage:    ./fetch_dataset.bash DATASET"
    echo ""
    echo "DATASET can be cacapo, cued, e2e, e2e-cleaned, e2e-enriched, neural-methodius, webnlg2023, or webnlg-enriched"
    echo ""
    }

if [ "$#" -lt 1 ]
then
    echo "Script requires at least one argument, a DATASET to fetch."
    echo ""
    usage
    exit 1
fi

DATASET=$1

if [ -v SUPPORTED_DATASETS["${DATASET}"] ]
then
  INITIAL_WD=$(pwd)
  echo "Fetching ${DATASET}..."
  mkdir -p datasets/raw
  cd datasets/raw || exit
  pwd
  fetch_dataset "${DATASET}"
  cd ${INITIAL_WD} || exit
  pwd
  echo "Postprocessing ${DATASET}"
  mkdir -p datasets/processed
  process_dataset "${DATASET}"
else
    echo ""
    echo "ERROR: No rule for fetching ${DATASET}"
    usage
    exit 1
fi

