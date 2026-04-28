#!/bin/bash

set -e

echo "Downloading kirsty_data.zip from Dropbox..."
curl -L -o kirsty_data.zip "https://www.dropbox.com/scl/fo/dypftzrojfv1r7neodl2c/AOCe623FNuep_elfBXuldXI?rlkey=9ywe6emm1nvajjke8oxvvpg1d&dl=1"

echo "Unzipping into kirsty_data/..."
unzip -o kirsty_data.zip -d kirsty_data

echo "Cleaning up zip file..."
rm kirsty_data.zip

echo "Done!"