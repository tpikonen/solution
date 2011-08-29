%.clu.ydat: %.p00.sub.ydat \
	    %.p01.sub.ydat \
            %.p02.sub.ydat \
            %.p03.sub.ydat \
            %.p04.sub.ydat \
            %.p05.sub.ydat \
            %.p06.sub.ydat \
            %.p07.sub.ydat \
            %.p08.sub.ydat \
            %.p09.sub.ydat
	cluster_positions -n $^

.PRECIOUS: s%.p00.fil.ydat
s%.p00.sub.ydat: s%.p00.fil.ydat
	subtract -s `echo $* | sed 's/^0*\(.*\)/\1/'` -i ./ -o ./ ../scans.yaml

s%.p00.fil.ydat: s%.mat
	filter_repetitions -m $^

bufs%.p00.out.ydat \
bufs%.p01.out.ydat \
bufs%.p02.out.ydat \
bufs%.p03.out.ydat \
bufs%.p04.out.ydat \
bufs%.p05.out.ydat \
bufs%.p06.out.ydat \
bufs%.p07.out.ydat \
bufs%.p08.out.ydat \
bufs%.p09.out.ydat: s%.mat
	buffer_filtering -n -m $^

.PRECIOUS: s%.mat
s%.mat:
	stackscans -m 10 -o ./ -c ../experiment_conf.yaml -s `echo $* | sed 's/^0*\(.*\)/\1/'` ../scans.yaml

depends.d: ../scans.yaml
	-rm $@
	$(eval scans = $(shell sed -n 's/[^0-9]*\([0-9][0-9]*\): *\[.*/\1/p' ../scans.yaml))
	for ss in $(scans) ; \
	do \
		bb=`sed -n "s/.*$${ss}:.*\[ *\(.*\) *,.*/\1/p" ../scans.yaml` ;\
		printf "s%03d.p00.sub.ydat: bufs%03d.p00.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p01.sub.ydat: bufs%03d.p01.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p02.sub.ydat: bufs%03d.p02.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p03.sub.ydat: bufs%03d.p03.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p04.sub.ydat: bufs%03d.p04.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p05.sub.ydat: bufs%03d.p05.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p06.sub.ydat: bufs%03d.p06.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p07.sub.ydat: bufs%03d.p07.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p08.sub.ydat: bufs%03d.p08.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p09.sub.ydat: bufs%03d.p09.out.ydat\n" $${ss} $${bb} >> $@ ; \
		printf "s%03d.p10.sub.ydat: bufs%03d.p10.out.ydat\n" $${ss} $${bb} >> $@ ; \
	done

include depends.d
