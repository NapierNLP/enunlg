#!/usr/bin/env bash
set -euo pipefail

declare -A SUPPORTED_DATASETS=()
for LABEL in cacapo cued e2e e2e-cleaned e2e-enriched neural-methodius webnlg2023 webnlg-enriched
do
  SUPPORTED_DATASETS["${LABEL}"]=1
done

DATASET=$1
echo "Fetching ${DATASET}..."

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
      git clone git@github.com:shawnwun/RNNLG.git ;;
    e2e)
      wget https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip
      unzip e2e-dataset.zip ;;
    e2e-cleaned)
      git clone git@github.com:tuetschek/e2e-cleaning.git ;;
    e2e-enriched)
      git clone git@github.com:ThiagoCF05/EnrichedE2E.git ;;
    neural-methodius)
      git clone git@github.com:aleksadre/methodiusNeuralINLG2021 ;;
    webnlg2023)
      git clone git@github.com:WebNLG/2023-Challenge.git ;;
    webnlg-enriched)
      git clone git@github.com:ThiagoCF05/webnlg.git ;;
  esac
}

if [[ ${SUPPORTED_DATASETS["${DATASET}"]} ]]
then
  mkdir -p datasets/raw
  cd datasets/raw || exit
  pwd
  fetch_dataset "${DATASET}"
  cd -
else
  echo "No rule for fetching ${DATASET}"
fi
