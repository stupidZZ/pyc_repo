#!/bin/bash

cd ./common/lib/bbox
/opt/conda/bin/python setup_linux.pyc build_ext --inplace
cd ../dataset/pycocotools
/opt/conda/bin/python setup_linux.pyc build_ext --inplace
cd ../../nms
/opt/conda/bin/python setup_linux.pyc build_ext --inplace
cd ../..
