echo "-- Cloning --"
git clone https://github.com/coastalcph/lex-glue
cd lex-glue
git checkout 20ad76a6d9da7794daddd94f23701c486ebad29f
echo "-- Applying patch --"
git apply --verbose ../0001-Update-experiment-file-to-work-without-cache_dir.patch

echo "-- Loading data --"
python3 ../download_data.py