docker run \
    --rm -ti \
    --network host \
    -e SERVER_ADDRESS='https://app.supervise.ly' \
    -e API_TOKEN='hpj70eBju05shIBsAzsO8c3jvouU7UgMne2xzc4Lrpg2Ad1VbOuveM63fsGrBUljfqVvSbQ9FsB5u2FNm7v62povH9bSYwaKUmaLU6SkzGuQI3kBPKVynnt7faQ5AdTO' \
    -v `pwd`/supervisely_lib:/workdir/supervisely_lib \
    -v `pwd`:/workdir/src \
    mwang029/supervisely_jupyter:latest