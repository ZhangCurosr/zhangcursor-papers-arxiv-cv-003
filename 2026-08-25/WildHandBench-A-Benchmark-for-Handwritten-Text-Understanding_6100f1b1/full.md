# WildHandBench: A Benchmark for Handwritten Text Understanding that Challenges MLLMs and Humans

Jun Zhang, Qiao Zhao, Cheng Cui, Jianying Qu, Zhongkai Sun, Jianwen Yang, Changda Zhou, ZhuoXin Liu, Shubin Han

B<sub>a</sub>id<sub>u</sub> I<sub>nc.</sub>

## Abstract

Whil<sub>e</sub> th<sub>e</sub> t<sub>op mo</sub>d<sub>e</sub>l <sub>on</sub> O<sub>mn</sub>iD<sub>oc</sub>B<sub>enc</sub>h <sub>now reac</sub>h<sub>es</sub> 96<sub>.</sub>34% <sub>overa</sub>ll <sub>on</sub> <sub>pr</sub>i<sub>n</sub>t<sub>e</sub>d<sub>-</sub>d<sub>ocumen</sub>t <sub>pars</sub>i<sub>ng,</sub> th<sub>e</sub> <sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>curren</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> t<sub>o</sub> h<sub>an</sub>dl<sub>e c</sub>h<sub>a</sub>ll<sub>eng</sub>i<sub>ng</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>s rema</sub>i<sub>ns</sub> l<sub>arge</sub>l<sub>y</sub> <sub>unc</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub>d<sub>.</sub> E<sub>x</sub>i<sub>s</sub>ti<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> f<sub>ocus</sub> <sub>on</sub> i<sub>so-</sub> l<sub>a</sub>t<sub>e</sub>d t<sub>ex</sub>t <sub>or</sub> f<sub>ormu</sub>l<sub>as, over</sub>l<sub>oo</sub>k h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> t<sub>a</sub>bl<sub>es an</sub>d <sub>rea</sub>l<sub>-</sub> <sub>wor</sub>ld d<sub>egra</sub>d<sub>a</sub>ti<sub>on, an</sub>d <sub>repor</sub>t <sub>aggrega</sub>t<sub>e accuracy w</sub>ith<sub>ou</sub>t <sub>ex-</sub> <sub>p</sub>l<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> <sub>w</sub>h<sub>y</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>a</sub>il<sub>.</sub>

We present WildHandBench, a benchmark containing 500 handwritten documents across three structures (free text, tables, formulas), four lan<sub>g</sub>ua<sub>g</sub>es, and nine real-world scenarios. We introduce a Prior-Driven Error (PDE) metric that quantifi<sub>es w</sub>h<sub>e</sub>th<sub>er errors or</sub>i<sub>g</sub>i<sub>na</sub>t<sub>e</sub> f<sub>rom</sub> l<sub>anguage pr</sub>i<sub>ors ra</sub>th<sub>er</sub> th<sub>an v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> 18 <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> to<sub>g</sub>ether with calibrated human baselines, we find: (1) the best model achieves only 71.85% overall; (2) humans outperform all models <sub>y</sub>et the <sub>g</sub>a<sub>p</sub> is narrow (77.09% vs. 71.85%); and (3) model errors are qualitativel<sub>y</sub> diferent from human <sub>errors—</sub>63<sub>–</sub>91% <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l <sub>errors are pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven versus on</sub>l<sub>y</sub> 49% f<sub>or</sub> h<sub>umans,</sub> <sub>expos</sub>i<sub>ng</sub> <sub>sys</sub>t<sub>ema</sub>ti<sub>c</sub> <sub>re</sub>li<sub>ance</sub> <sub>on</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>ors</sub> th<sub>a</sub>t <sub>conven</sub>ti<sub>ona</sub>l <sub>accuracy</sub> <sub>me</sub>t<sub>r</sub>i<sub>cs</sub> <sub>canno</sub>t <sub>cap</sub>t<sub>ure.</sub>

## Introduction

R<sub>ecen</sub>t <sub>a</sub>d<sub>vances</sub> i<sub>n mu</sub>lti<sub>mo</sub>d<sub>a</sub>l l<sub>arge</sub> l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s</sub> (MLLMs) have dramaticall<sub>y</sub> im<sub>p</sub>roved document understandin<sub>g</sub> (Fu et al. 2025; Ou<sub>y</sub>an<sub>g</sub> et al. 2025). The to<sub>p</sub> model <sub>on</sub> th<sub>e</sub> O<sub>mn</sub>iD<sub>oc</sub>B<sub>enc</sub>h l<sub>ea</sub>d<sub>er</sub>b<sub>oar</sub>d <sub>now reac</sub>h<sub>es</sub> 96<sub>.</sub>34% overall (Ou<sub>y</sub>an<sub>g</sub> et al. 2025), a<sub>pp</sub>roachin<sub>g</sub> saturation on <sub>pr</sub>i<sub>n</sub>t<sub>e</sub>d<sub>-</sub>d<sub>ocumen</sub>t <sub>pars</sub>i<sub>ng.</sub> Th<sub>ese remar</sub>k<sub>a</sub>bl<sub>e resu</sub>lt<sub>s</sub> h<sub>ave</sub> <sub>gra</sub>d<sub>ua</sub>ll<sub>y crea</sub>t<sub>e</sub>d <sub>a w</sub>id<sub>esprea</sub>d <sub>percep</sub>ti<sub>on</sub> th<sub>a</sub>t OCR i<sub>s</sub> b<sub>e-</sub> <sub>com</sub>i<sub>ng a</sub> l<sub>arge</sub>l<sub>y so</sub>l<sub>ve</sub>d <sub>pro</sub>bl<sub>em.</sub>

W<sub>e</sub> <sub>argue</sub> th<sub>a</sub>t thi<sub>s</sub> <sub>conc</sub>l<sub>us</sub>i<sub>on</sub> i<sub>s</sub> <sub>prema</sub>t<sub>ure.</sub>

N<sub>ear</sub>l<sub>y a</sub>ll <sub>ev</sub>id<sub>ence suppor</sub>ti<sub>ng</sub> thi<sub>s percep</sub>ti<sub>on comes</sub> f<sub>rom</sub> <sub>pr</sub>i<sub>n</sub>t<sub>e</sub>d d<sub>ocumen</sub>t<sub>s</sub> <sub>w</sub>ith <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>regu</sub>l<sub>ar</sub> l<sub>ayou</sub>t<sub>s</sub> <sub>an</sub>d <sub>c</sub>l<sub>ear</sub> visual a<sub>pp</sub>earances (Mathew, Karatzas, and Jawahar 2021; Zh<sub>ong,</sub> Sh<sub>a</sub>fi<sub>e</sub>iB<sub>avan</sub>i<sub>, an</sub>d Ji<sub>meno</sub> Y<sub>epes</sub> 2020<sub>;</sub> Zh<sub>eng e</sub>t <sub>a</sub>l<sub>.</sub> 2021; Göbel et al. 2013). Handwritten documents tell a diff<sub>eren</sub>t <sub>s</sub>t<sub>ory.</sub> M<sub>e</sub>di<sub>ca</sub>l <sub>recor</sub>d<sub>s,</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> f<sub>orms, c</sub>l<sub>assroom</sub> <sub>no</sub>t<sub>es,</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>l <sub>arc</sub>hi<sub>ves,</sub> <sub>an</sub>d <sub>persona</sub>l <sub>correspon</sub>d<sub>ence</sub> <sub>re-</sub> <sub>ma</sub>i<sub>n su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y more</sub> difi<sub>cu</sub>lt f<sub>or</sub> t<sub>o</sub>d<sub>ay</sub>’<sub>s</sub> MLLM<sub>s</sub> d<sub>e-</sub> <sub>sp</sub>it<sub>e</sub> th<sub>e</sub>i<sub>r</sub> i<sub>mpress</sub>i<sub>ve per</sub>f<sub>ormance on pr</sub>i<sub>n</sub>t<sub>e</sub>d d<sub>ocumen</sub>t<sub>s.</sub> U<sub>n</sub>lik<sub>e</sub> <sub>pr</sub>i<sub>n</sub>t<sub>e</sub>d OCR<sub>,</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> requires models to jointly resolve ambiguous handwriting, <sub>recover</sub> i<sub>rregu</sub>l<sub>ar</sub> d<sub>ocumen</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ures,</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h <sub>v</sub>i<sub>sua</sub>l <sub>ev-</sub> id<sub>ence</sub> f<sub>rom</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>ors,</sub> <sub>an</sub>d <sub>reason</sub> <sub>un</sub>d<sub>er</sub> <sub>genu</sub>i<sub>ne</sub> <sub>un-</sub> certaint<sub>y</sub> (Guan et al. 2024; Bai et al. 2024). Consequentl<sub>y</sub>, h<sub>an</sub>d<sub>wr</sub>iti<sub>ng</sub> i<sub>s no</sub>t <sub>s</sub>i<sub>mp</sub>l<sub>y a more</sub> difi<sub>cu</sub>lt OCR <sub>pro</sub>bl<sub>em,</sub> b<sub>u</sub>t <sub>a</sub> di<sub>s</sub>ti<sub>nc</sub>t d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> <sub>c</sub>h<sub>a</sub>ll<sub>enge.</sub>

S<sub>urpr</sub>i<sub>s</sub>i<sub>ng</sub>l<sub>y, curren</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s prov</sub>id<sub>e</sub> li<sub>m-</sub> it<sub>e</sub>d i<sub>ns</sub>i<sub>g</sub>ht i<sub>n</sub>t<sub>o w</sub>h<sub>y</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> <sub>rema</sub>i<sub>ns unso</sub>l<sub>ve</sub>d<sub>.</sub> E<sub>x</sub>i<sub>s</sub>ti<sub>ng</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s pr</sub>i<sub>mar-</sub> il<sub>y eva</sub>l<sub>ua</sub>t<sub>e</sub> i<sub>so</sub>l<sub>a</sub>t<sub>e</sub>d f<sub>ree-</sub>t<sub>ex</sub>t <sub>recogn</sub>iti<sub>on, w</sub>hil<sub>e</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> t<sub>a</sub>bl<sub>es, na</sub>t<sub>ura</sub>ll<sub>y wr</sub>itt<sub>en</sub> f<sub>ormu</sub>l<sub>as, an</sub>d <sub>rea</sub>l<sub>-wor</sub>ld d<sub>ocumen</sub>t d<sub>egra</sub>d<sub>a</sub>ti<sub>on rema</sub>i<sub>n</sub> l<sub>arge</sub>l<sub>y unexp</sub>l<sub>ore</sub>d<sub>.</sub> M<sub>ore</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>ll<sub>y,</sub> <sub>ex</sub>i<sub>s</sub>ti<sub>ng eva</sub>l<sub>ua</sub>ti<sub>ons</sub> f<sub>ocus a</sub>l<sub>mos</sub>t <sub>exc</sub>l<sub>us</sub>i<sub>ve</sub>l<sub>y on recogn</sub>iti<sub>on</sub> accurac<sub>y</sub> (Fu et al. 2025; Ou<sub>y</sub>an<sub>g</sub> et al. 2025). The<sub>y</sub> measure how often models fail, but provide little understanding of why th<sub>ey</sub> f<sub>a</sub>il<sub>—</sub>i<sub>n par</sub>ti<sub>cu</sub>l<sub>ar, w</sub>h<sub>e</sub>th<sub>er errors or</sub>i<sub>g</sub>i<sub>na</sub>t<sub>e</sub> f<sub>rom poor</sub> <sub>v</sub>i<sub>sua</sub>l <sub>percep</sub>ti<sub>on or</sub> f<sub>rom excess</sub>i<sub>ve re</sub>li<sub>ance on</sub> l<sub>anguage pr</sub>i<sub>-</sub> ors.

To address this gap, we introduce WildHandBench, a b<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng.</sub> W<sub>ild-</sub> H<sub>and</sub>B<sub>ench</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> 500 h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>s</sub> <sub>spann</sub>i<sub>ng</sub> three document structures (free text, tables, and formulas), f<sub>our</sub> l<sub>anguage</sub> <sub>se</sub>tti<sub>ngs,</sub> <sub>an</sub>d <sub>n</sub>i<sub>ne</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>ve</sub> <sub>rea</sub>l<sub>-wor</sub>ld scenarios. We also introduce a Prior-Driven Error (PDE) <sub>me</sub>t<sub>r</sub>i<sub>c</sub> th<sub>a</sub>t <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>zes w</sub>h<sub>e</sub>th<sub>er mo</sub>d<sub>e</sub>l <sub>errors ar</sub>i<sub>se</sub> f<sub>rom</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>ors</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence,</sub> <sub>ena</sub>bli<sub>ng</sub> <sub>sys</sub>t<sub>em-</sub> <sub>a</sub>ti<sub>c ana</sub>l<sub>ys</sub>i<sub>s o</sub>f h<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>ti<sub>on-</sub>lik<sub>e recogn</sub>iti<sub>on</sub> b<sub>e</sub>h<sub>av</sub>i<sub>ors.</sub>

U<sub>s</sub>in<sub>g</sub> W<sub>ild</sub>H<sub>and</sub>B<sub>ench, we co</sub>nd<sub>uc</sub>t <sub>a co</sub>m<sub>p</sub>r<sub>e</sub>h<sub>e</sub>n<sub>s</sub>i<sub>ve</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on o</sub>f 18 <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t MLLM<sub>s an</sub>d OCR<sub>-or</sub>i<sub>en</sub>t<sub>e</sub>d <sub>v</sub>i<sub>s</sub>i<sub>on-</sub>l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s un</sub>d<sub>er a un</sub>ifi<sub>e</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on pro</sub>t<sub>oco</sub>l t<sub>oge</sub>th<sub>er w</sub>ith <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d h<sub>uman</sub> b<sub>ase</sub>li<sub>nes.</sub> O<sub>ur s</sub>t<sub>u</sub>d<sub>y y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>ree</sub> k<sub>ey</sub> fi<sub>n</sub>di<sub>ngs.</sub> Fi<sub>rs</sub>t<sub>,</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>d<sub>-</sub> i<sub>ng rema</sub>i<sub>ns</sub> f<sub>ar</sub> f<sub>rom so</sub>l<sub>ve</sub>d<sub>:</sub> th<sub>e s</sub>t<sub>ronges</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>ac</sub>hi<sub>eves</sub> <sub>on</sub>l<sub>y</sub> 71<sub>.</sub>85% <sub>overa</sub>ll<sub>.</sub> S<sub>econ</sub>d<sub>,</sub> h<sub>umans ou</sub>t<sub>per</sub>f<sub>orm a</sub>ll <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> (77.09% vs. 71.85%), <sub>y</sub>et the <sub>g</sub>a<sub>p</sub> is narrow—to<sub>p</sub> models even <sub>surpass</sub> h<sub>umans on</sub> t<sub>ex</sub>t t<sub>ranscr</sub>i<sub>p</sub>ti<sub>on un</sub>d<sub>er</sub> f<sub>orma</sub>t<sub>-ma</sub>t<sub>c</sub>hi<sub>ng</sub> <sub>me</sub>t<sub>r</sub>i<sub>cs.</sub> Thi<sub>r</sub>d<sub>, mo</sub>d<sub>e</sub>l <sub>errors are qua</sub>lit<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> dif<sub>eren</sub>t f<sub>rom</sub> h<sub>uman errors:</sub> h<sub>umans rema</sub>i<sub>n conserva</sub>ti<sub>ve on</sub> ill<sub>eg</sub>ibl<sub>e con-</sub> t<sub>en</sub>t<sub>, w</sub>h<sub>ereas mo</sub>d<sub>e</sub>l<sub>s con</sub>fid<sub>en</sub>tl<sub>y</sub> h<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>t<sub>e</sub> fl<sub>uen</sub>t b<sub>u</sub>t <sub>un-</sub> <sub>suppor</sub>t<sub>e</sub>d t<sub>ex</sub>t<sub>, w</sub>ith 63<sub>–</sub>91% <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l <sub>errors c</sub>l<sub>ass</sub>ifi<sub>e</sub>d <sub>as</sub> <sub>pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven compare</sub>d t<sub>o on</sub>l<sub>y</sub> 49% f<sub>or</sub> h<sub>umans.</sub> Thi<sub>s exposes</sub> <sub>sys</sub>t<sub>ema</sub>ti<sub>c</sub> <sub>re</sub>li<sub>ance</sub> <sub>on</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>ors</sub> th<sub>a</sub>t <sub>conven</sub>ti<sub>ona</sub>l <sub>ac-</sub> <sup>c</sup>u<sup>rac</sup>y <sup>metrics</sup> <sup>cannot</sup> <sup>ca</sup>p<sup>t</sup>u<sup>re</sup>.

O<sub>ur con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons are summar</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

• We <sub>p</sub>ro<sub>p</sub>ose WildHandBench<sub>,</sub> to our knowled<sub>g</sub>e the first benchmark that jointly evaluates handwritten free text, tabl<sub>es, an</sub>d f<sub>ormu</sub>l<sub>as across mu</sub>lti<sub>p</sub>l<sub>e</sub> l<sub>anguages an</sub>d di<sub>verse</sub> <sub>rea</sub>l<sub>-wor</sub>ld <sub>scenar</sub>i<sub>os,</sub> <sub>a</sub>dd<sub>ress</sub>i<sub>ng</sub> k<sub>ey</sub> <sub>gaps</sub> i<sub>n</sub> <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> <sub>eva</sub>l<sub>-</sub> uat<sup>i</sup>on covera<sub>g</sub>e.

• We introduce the Prior-Driven Error (PDE) metric<sub>,</sub> which <sub>quan</sub>tifi<sub>es w</sub>h<sub>e</sub>th<sub>er mo</sub>d<sub>e</sub>l <sub>errors or</sub>i<sub>g</sub>i<sub>na</sub>t<sub>e</sub> f<sub>rom</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>ors ra</sub>th<sub>er</sub> th<sub>an v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence, ena</sub>bli<sub>ng sys</sub>t<sub>ema</sub>ti<sub>c</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s o</sub>f h<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>ti<sub>on-</sub>lik<sub>e recogn</sub>iti<sub>on</sub> b<sub>e</sub>h<sub>av</sub>i<sub>ors</sub> b<sub>e-</sub> <sub>yon</sub>d <sub>conven</sub>ti<sub>ona</sub>l <sub>accuracy</sub> <sub>me</sub>t<sub>r</sub>i<sub>cs.</sub>

• We conduct a com<sub>p</sub>rehensive evaluation of 18 state-ofth<sub>e-a</sub>rt MLLM<sub>s a</sub>nd OCR<sub>-o</sub>ri<sub>e</sub>nt<sub>e</sub>d <sub>v</sub>i<sub>s</sub>i<sub>o</sub>n<sub>-</sub>l<sub>a</sub>n<sub>guage</sub> m<sub>o</sub>d<sub>-</sub> <sub>e</sub>l<sub>s</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d h<sub>uman</sub> b<sub>ase</sub>li<sub>nes,</sub> <sub>revea</sub>li<sub>ng</sub> th<sub>e</sub> <sub>pers</sub>i<sub>s</sub>t<sub>en</sub>t h<sub>an</sub>d<sub>wr</sub>iti<sub>ng</sub> <sub>gap,</sub> th<sub>e</sub> <sub>narrow</sub> <sub>ye</sub>t <sub>mean</sub>i<sub>ng-</sub> f<sub>u</sub>l di<sub>s</sub>t<sub>ance</sub> t<sub>o</sub> h<sub>uman per</sub>f<sub>ormance, an</sub>d th<sub>e sys</sub>t<sub>ema</sub>ti<sub>c</sub> r<sub>e</sub>li<sub>a</sub>n<sub>ce</sub> <sub>o</sub>f MLLM<sub>s</sub> <sub>o</sub>n l<sub>a</sub>n<sub>guage</sub> <sub>p</sub>ri<sub>o</sub>r<sub>s.</sub>

## Related Work

Handwritten document understanding. Handwritten doc-<sub>umen</sub>t <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> h<sub>as</sub> b<sub>een s</sub>t<sub>u</sub>di<sub>e</sub>d f<sub>or</sub> d<sub>eca</sub>d<sub>es, ye</sub>t <sub>ex-</sub> i<sub>s</sub>ti<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s rema</sub>i<sub>n</sub> hi<sub>g</sub>hl<sub>y</sub> t<sub>as</sub>k<sub>-spec</sub>ifi<sub>c an</sub>d f<sub>rag-</sub> <sub>men</sub>t<sub>e</sub>d<sub>.</sub> M<sub>os</sub>t d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> f<sub>ocus on</sub> li<sub>ne-</sub>l<sub>eve</sub>l h<sub>an</sub>d<sub>wr</sub>it<sub>-</sub> ten text reco<sub>g</sub>nition, includin<sub>g</sub> IAM (Marti and Bunke 2002), RIMES (Grosicki et al. 2009), CASIA-HWDB (Liu et al. 2011), SCUT-EPT (Zhu et al. 2019), and SCUT-HCCDoc (Zhan<sub>g</sub>, Lian<sub>g</sub>, and Jin 2020), coverin<sub>g</sub> Latin and Chi<sub>nese</sub> h<sub>an</sub>d<sub>wr</sub>iti<sub>ng un</sub>d<sub>er re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y con</sub>t<sub>ro</sub>ll<sub>e</sub>d <sub>se</sub>tti<sub>ngs.</sub> S<sub>c</sub>i<sub>en</sub>tifi<sub>c</sub> h<sub>an</sub>d<sub>wr</sub>iti<sub>ng</sub> h<sub>as</sub> b<sub>een</sub> i<sub>nves</sub>ti<sub>ga</sub>t<sub>e</sub>d th<sub>roug</sub>h d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> such as NoTeS-Bank (Pal et al. 2025), while handwritten <sub>ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l <sub>express</sub>i<sub>on</sub> <sub>recogn</sub>iti<sub>on</sub> i<sub>s</sub> <sub>pr</sub>i<sub>mar</sub>il<sub>y</sub> b<sub>enc</sub>h<sub>-</sub> marked b<sub>y</sub> CROHME (Mahdavi et al. 2019) and MathWritin<sub>g</sub> (Gervais, Fadeeva, and Maksai 2024). In contrast, table understandin<sub>g</sub> benchmarks, includin<sub>g</sub> PubTabNet (Zhon<sub>g</sub>, ShafieiBavani, and Jimeno Ye<sub>p</sub>es 2020), FinTabNet (Zhen<sub>g</sub> et al. 2021), and the ICDAR table com etitions (Göbel et al. 2013), are exclusivel<sub>y</sub> constructed from <sub>p</sub>rinted documents. Alth<sub>oug</sub>h th<sub>ese</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> h<sub>ave s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y a</sub>d<sub>vance</sub>d i<sub>n</sub>di<sub>-</sub> <sub>v</sub>id<sub>ua</sub>l OCR t<sub>as</sub>k<sub>s,</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub> t<sub>ex</sub>t<sub>,</sub> f<sub>ormu</sub>l<sub>as, an</sub>d t<sub>a</sub>bl<sub>es</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>an</sub>d <sub>prov</sub>id<sub>e</sub> li<sub>m</sub>it<sub>e</sub>d <sub>coverage</sub> <sub>o</sub>f <sub>c</sub>h<sub>a</sub>ll<sub>eng</sub>i<sub>ng</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>s w</sub>ith <sub>rea</sub>l<sub>-wor</sub>ld d<sub>egra</sub>d<sub>a</sub>ti<sub>on.</sub> C<sub>onse-</sub> <sub>quen</sub>tl<sub>y, curren</sub>t h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s rema</sub>i<sub>n</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>t f<sub>or eva</sub>l<sub>ua</sub>ti<sub>ng</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>s as a un</sub>ifi<sub>e</sub>d d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng pro</sub>bl<sub>em.</sub>

Benchmarking document understanding. The rapid develo<sub>p</sub>ment of multimodal lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models (MLLMs) h<sub>as s</sub>hift<sub>e</sub>d d<sub>ocumen</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> f<sub>rom</sub> i<sub>so</sub>l<sub>a</sub>t<sub>e</sub>d OCR t<sub>as</sub>k<sub>s</sub> t<sub>owar</sub>d h<sub>o</sub>li<sub>s</sub>ti<sub>c</sub> d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng.</sub> B<sub>enc</sub>h<sub>mar</sub>k<sub>s suc</sub>h as OmniDocBench (Ou<sub>y</sub>an<sub>g</sub> et al. 2025) evaluate diverse d<sub>ocumen</sub>t <sub>pars</sub>i<sub>ng capa</sub>biliti<sub>es un</sub>d<sub>er a un</sub>ifi<sub>e</sub>d f<sub>ramewor</sub>k<sub>,</sub> while Real5-OmniDocBench (Zhou et al. 2026) further in-<sub>ves</sub>ti<sub>ga</sub>t<sub>es ro</sub>b<sub>us</sub>t<sub>ness un</sub>d<sub>er rea</sub>l<sub>-wor</sub>ld <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l d<sub>egra</sub>d<sub>a</sub>ti<sub>ons</sub> throu<sub>g</sub>h fine-<sub>g</sub>rained factor-level evaluation. OCRBench (Liu et al. 2024) and OCRBench v2 (Fu et al. 2025) extend bench-<sub>mar</sub>k <sub>coverage</sub> t<sub>o</sub> OCR<sub>-or</sub>i<sub>en</sub>t<sub>e</sub>d <sub>mu</sub>lti<sub>mo</sub>d<sub>a</sub>l <sub>capa</sub>biliti<sub>es,</sub> i<sub>n-</sub> <sub>c</sub>l<sub>u</sub>di<sub>ng</sub> t<sub>ex</sub>t <sub>recogn</sub>iti<sub>on,</sub> f<sub>ormu</sub>l<sub>a</sub> <sub>pars</sub>i<sub>ng,</sub> t<sub>a</sub>bl<sub>e</sub> <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>d<sub>-</sub> i<sub>ng, an</sub>d d<sub>ocumen</sub>t <sub>v</sub>i<sub>sua</sub>l <sub>ques</sub>ti<sub>on answer</sub>i<sub>ng.</sub> Th<sub>ese</sub> b<sub>enc</sub>h<sub>-</sub> <sub>mar</sub>k<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> <sub>remar</sub>k<sub>a</sub>bl<sub>e</sub> <sub>progress</sub> <sub>on</sub> <sub>pr</sub>i<sub>n</sub>t<sub>e</sub>d d<sub>ocu-</sub> <sub>men</sub>t<sub>s, w</sub>ith th<sub>e</sub> t<sub>op mo</sub>d<sub>e</sub>l <sub>now reac</sub>hi<sub>ng</sub> 96<sub>.</sub>34% <sub>overa</sub>ll <sub>on</sub> OmniDocBench (Ou<sub>y</sub>an<sub>g</sub> et al. 2025). Nevertheless, these <sup>e</sup>v<sup>al</sup>u<sup>ations</sup> p<sup>rimaril</sup>y <sup>re</sup>p<sup>ort</sup> <sup>a</sup>gg<sup>re</sup>g<sup>ate</sup> <sup>reco</sup>g<sup>nition</sup> <sup>acc</sup>u<sup>rac</sup>y, <sub>prov</sub>idi<sub>ng</sub> li<sub>m</sub>it<sub>e</sub>d <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> <sub>o</sub>f <sub>w</sub>h<sub>y</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocu-</sub> <sub>men</sub>t<sub>s rema</sub>i<sub>n su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y more c</sub>h<sub>a</sub>ll<sub>eng</sub>i<sub>ng</sub> th<sub>an pr</sub>i<sub>n</sub>t<sub>e</sub>d d<sub>ocumen</sub>t<sub>s.</sub>

Prior-driven errors in vision-language models. Re-<sub>cen</sub>t <sub>s</sub>t<sub>u</sub>di<sub>es</sub> h<sub>ave</sub> <sub>s</sub>h<sub>own</sub> th<sub>a</sub>t <sub>mu</sub>lti<sub>mo</sub>d<sub>a</sub>l l<sub>arge</sub> l<sub>anguage</sub> models (MLLMs) frequentl<sub>y</sub> <sub>g</sub>enerate out<sub>p</sub>uts that are not full<sub>y g</sub>rounded in visual evidence (Guan et al. 2024; Bai et al. 2024). In OCR-related tasks, such hallucinations oft<sub>en</sub> <sub>man</sub>if<sub>es</sub>t <sub>as</sub> fl<sub>uen</sub>t <sub>ye</sub>t <sub>v</sub>i<sub>sua</sub>ll<sub>y</sub> <sub>unsuppor</sub>t<sub>e</sub>d t<sub>ranscr</sub>i<sub>p-</sub> ti<sub>ons, w</sub>h<sub>ere</sub> l<sub>anguage pr</sub>i<sub>ors overr</sub>id<sub>e am</sub>bi<sub>guous v</sub>i<sub>sua</sub>l <sub>o</sub>b<sub>-</sub> <sub>serva</sub>ti<sub>ons.</sub> T<sub>o</sub> b<sub>e</sub>tt<sub>er c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> thi<sub>s p</sub>h<sub>enomenon,</sub> S<sub>eong</sub> et al. (Seong et al. 2026) proposed PINK, which penali<sub>zes over-correc</sub>ti<sub>on</sub> i<sub>n</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l <sub>express</sub>i<sub>on</sub> reco<sub>g</sub>nition. Similarl<sub>y</sub>, Hun<sub>y</sub>uanOCR-1.5 (Li et al. 2026) i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> CHAOS<sub>-</sub>B<sub>enc</sub>h<sub>, w</sub>hi<sub>c</sub>h <sub>eva</sub>l<sub>ua</sub>t<sub>es w</sub>h<sub>e</sub>th<sub>er mo</sub>d<sub>-</sub> <sub>e</sub>l<sub>s</sub> f<sub>a</sub>ithf<sub>u</sub>ll<sub>y preserve v</sub>i<sub>sua</sub>ll<sub>y o</sub>b<sub>serve</sub>d b<sub>u</sub>t <sub>seman</sub>ti<sub>ca</sub>ll<sub>y</sub> i<sub>mp</sub>l<sub>aus</sub>ibl<sub>e</sub> t<sub>ex</sub>t <sub>un</sub>d<sub>er syn</sub>th<sub>e</sub>ti<sub>c c</sub>h<sub>arac</sub>t<sub>er per</sub>t<sub>ur</sub>b<sub>a</sub>ti<sub>ons</sub> i<sub>n</sub> <sub>pr</sub>i<sub>n</sub>t<sub>e</sub>d d<sub>ocumen</sub>t<sub>s.</sub> Th<sub>ese s</sub>t<sub>u</sub>di<sub>es cons</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t <sub>pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven recogn</sub>iti<sub>on errors are sys</sub>t<sub>ema</sub>ti<sub>c ra</sub>th<sub>er</sub> th<sub>an</sub> i<sub>nc</sub>id<sub>en</sub>t<sub>a</sub>l<sub>.</sub>

H<sub>owever, ex</sub>i<sub>s</sub>ti<sub>ng eva</sub>l<sub>ua</sub>ti<sub>ons rema</sub>i<sub>n res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o e</sub>ith<sub>er</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l <sub>express</sub>i<sub>ons or syn</sub>th<sub>e</sub>ti<sub>ca</sub>ll<sub>y per-</sub> t<sub>ur</sub>b<sub>e</sub>d <sub>pr</sub>i<sub>n</sub>t<sub>e</sub>d d<sub>ocumen</sub>t<sub>s.</sub> T<sub>o</sub> th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>f <sub>our</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge, no</sub> <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>sys</sub>t<sub>ema</sub>ti<sub>ca</sub>ll<sub>y measures pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven er-</sub> <sub>rors</sub> <sub>on</sub> <sub>na</sub>t<sub>ura</sub>ll<sub>y</sub> <sub>occurr</sub>i<sub>ng</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>s</sub> <sub>spann</sub>i<sub>ng</sub> f<sub>ree</sub> t<sub>ex</sub>t<sub>,</sub> t<sub>a</sub>bl<sub>es, an</sub>d f<sub>ormu</sub>l<sub>as.</sub> O<sub>ur propose</sub>d P<sub>r</sub>i<sub>or-</sub>D<sub>r</sub>i<sub>ven</sub> Error (PDE) metric is desi<sub>g</sub>ned to brid<sub>g</sub>e this <sub>g</sub>a<sub>p</sub> b<sub>y</sub> quantif<sub>y</sub>i<sub>ng</sub> th<sub>e ex</sub>t<sub>en</sub>t t<sub>o w</sub>hi<sub>c</sub>h <sub>mo</sub>d<sub>e</sub>l <sub>errors are a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>a</sub>bl<sub>e</sub> t<sub>o</sub> l<sub>anguage pr</sub>i<sub>ors</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence</sub> i<sub>n rea</sub>li<sub>s</sub>ti<sub>c</sub> h<sub>an</sub>d<sub>-</sub> <sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng.</sub>

T<sub>a</sub>bl<sub>e</sub> 1 <sub>summar</sub>i<sub>zes represen</sub>t<sub>a</sub>ti<sub>ve</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub> C<sub>om-</sub> <sub>pare</sub>d <sub>w</sub>ith <sub>prev</sub>i<sub>ous</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s,</sub> W<sub>ild</sub>H<sub>and</sub>B<sub>ench</sub> i<sub>s,</sub> t<sub>o our</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge,</sub> th<sub>e</sub> fi<sub>rs</sub>t t<sub>o un</sub>if<sub>y</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> f<sub>ree</sub> t<sub>ex</sub>t<sub>,</sub> t<sub>a</sub>bl<sub>es,</sub> <sub>an</sub>d f<sub>ormu</sub>l<sub>as w</sub>ithi<sub>n a s</sub>i<sub>ng</sub>l<sub>e eva</sub>l<sub>ua</sub>ti<sub>on</sub> f<sub>ramewor</sub>k <sub>w</sub>ith <sub>ca</sub>l<sub>-</sub> ib<sub>ra</sub>t<sub>e</sub>d h<sub>uman</sub> b<sub>ase</sub>li<sub>nes an</sub>d P<sub>r</sub>i<sub>or-</sub>D<sub>r</sub>i<sub>ven</sub> E<sub>rror ana</sub>l<sub>ys</sub>i<sub>s.</sub>

## WildHandBench: Benchmark Design

WildHandBench jointly evaluates handwritten free text, t<sub>a</sub>bl<sub>es, an</sub>d f<sub>ormu</sub>l<sub>as un</sub>d<sub>er rea</sub>li<sub>s</sub>ti<sub>c</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>scenar</sub>i<sub>os.</sub> Th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> 500 h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocu-</sub> <sub>men</sub>t i<sub>mages spann</sub>i<sub>ng</sub> th<sub>ree</sub> d<sub>ocumen</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ures,</sub> f<sub>our</sub> l<sub>an-</sub> <sub>guage</sub> <sub>se</sub>tti<sub>ngs,</sub> <sub>an</sub>d <sub>n</sub>i<sub>ne</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>ve</sub> <sub>rea</sub>l<sub>-wor</sub>ld <sub>scenar</sub>i<sub>os.</sub> Fi<sub>gure</sub> 1 ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>e overa</sub>ll <sub>cons</sub>t<sub>ruc</sub>ti<sub>on p</sub>i<sub>pe</sub>li<sub>ne.</sub>

## Benchmark Taxonomy

T<sub>o compre</sub>h<sub>ens</sub>i<sub>ve</sub>l<sub>y c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>un-</sub> d<sub>ers</sub>t<sub>an</sub>di<sub>ng, every samp</sub>l<sub>e</sub> i<sub>s organ</sub>i<sub>ze</sub>d <sub>a</sub>l<sub>ong</sub> th<sub>ree</sub> di<sub>men-</sub> <sub>s</sub>i<sub>ons.</sub>

• Language (4). WildHandBench covers four lan<sub>g</sub>ua<sub>g</sub>e settin<sub>g</sub>s, includin<sub>g</sub> Sim<sub>p</sub>lified Chinese (333, 66.6%), En-<sub>g</sub>lish (143, 28.6%), Traditional Chinese (13, 2.6%), and Chinese–En<sub>g</sub>lish mixed documents (11, 2.2%).

• Document Structure (3). The benchmark jointl<sub>y</sub> eval-<sub>ua</sub>t<sub>es</sub> th<sub>ree represen</sub>t<sub>a</sub>ti<sub>ve</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>s</sub>t<sub>ruc-</sub> tures: free text (367, 73.4%), tables (81, 16.2%), and formulas (52 ima<sub>g</sub>es containin<sub>g</sub> 278 individuall<sub>y</sub> annotated formula re<sub>g</sub>ions, 10.4%).

<table><tr><td>Benchmark</td><td>Lang.</td><td>Handwritten</td><td>Free Text</td><td>Table</td><td>Formula</td><td>Real Degrad.</td><td>Human Baseline</td></tr><tr><td>IAM</td><td>EN</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>CASIA-HWDB</td><td>ZH</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>SCUT-EPT / HCCDoc</td><td>ZH</td><td>√</td><td>√</td><td></td><td></td><td>Partial</td><td></td></tr><tr><td>CROHME / MathWriting</td><td>Symbol</td><td>√</td><td>一</td><td></td><td>√</td><td></td><td></td></tr><tr><td>PubTabNet / FinTabNet</td><td>EN</td><td></td><td></td><td>√</td><td>一</td><td></td><td></td></tr><tr><td>ICDAR Table</td><td>EN</td><td></td><td>一</td><td>√</td><td>一</td><td></td><td></td></tr><tr><td>OCRBench (v1/v2)</td><td>Multi</td><td>Partial</td><td>√</td><td>√</td><td>√</td><td>Partial</td><td></td></tr><tr><td>OmniDocBench</td><td>ZH/EN</td><td>Partial</td><td>√</td><td>√</td><td>√</td><td>Partial</td><td></td></tr><tr><td>Real5-OmniDocBench</td><td>ZH/EN</td><td>Partial</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>WILDHANDBENCH (Ours)</td><td>ZH/EN</td><td>了</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Comparison with representative benchmarks. WildHandBench is, to our knowledge, the first to jointly cover hand <sub>wr</sub>itt<sub>en</sub> f<sub>ree</sub> t<sub>ex</sub>t<sub>,</sub> t<sub>a</sub>bl<sub>es, an</sub>d f<sub>ormu</sub>l<sub>as w</sub>ith <sub>rea</sub>l<sub>-scenar</sub>i<sub>o</sub> d<sub>egra</sub>d<sub>a</sub>ti<sub>on an</sub>d <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d h<sub>uman</sub> b<sub>ase</sub>li<sub>nes.</sub>

• Real-world Scenario (9). Sam<sub>p</sub>les are collected from <sub>n</sub>i<sub>ne represen</sub>t<sub>a</sub>ti<sub>ve scenar</sub>i<sub>os,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> lit<sub>erary wr</sub>iti<sub>ng,</sub> l<sub>e</sub>tt<sub>ers an</sub>d <sub>no</sub>t<sub>es, me</sub>di<sub>ca</sub>l <sub>recor</sub>d<sub>s,</sub> b<sub>us</sub>i<sub>ness an</sub>d <sub>govern-</sub> <sub>men</sub>t d<sub>ocumen</sub>t<sub>s, e</sub>d<sub>uca</sub>ti<sub>on, ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l <sub>an</sub>d <sub>sc</sub>i<sub>en</sub>tifi<sub>c</sub> <sub>ma</sub>t<sub>er</sub>i<sub>a</sub>l<sub>s, c</sub>l<sub>ass</sub>i<sub>ca</sub>l <sub>ca</sub>lli<sub>grap</sub>h<sub>y,</sub> d<sub>a</sub>il<sub>y m</sub>i<sub>sce</sub>ll<sub>any, an</sub>d hi<sub>s-</sub> t<sub>or</sub>i<sub>ca</sub>l <sub>arc</sub>hi<sub>ves.</sub>

## Benchmark Construction

T<sub>o max</sub>i<sub>m</sub>i<sub>ze</sub> b<sub>o</sub>th d<sub>a</sub>t<sub>ase</sub>t di<sub>vers</sub>it<sub>y an</sub>d <sub>anno</sub>t<sub>a</sub>ti<sub>on re</sub>li<sub>a</sub>bilit<sub>y,</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>s</sub> <sub>are</sub> <sub>co</sub>ll<sub>ec</sub>t<sub>e</sub>d f<sub>rom</sub> t<sub>wo</sub> <sub>comp</sub>l<sub>emen-</sub> <sup>tar</sup>y <sup>sources.</sup>

Ofline handwritten manuscripts. Participants voluntaril<sub>y con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e au</sub>th<sub>en</sub>ti<sub>c</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en ma</sub>t<sub>er</sub>i<sub>a</sub>l<sub>s cover</sub>i<sub>ng mu</sub>l<sub>-</sub> ti<sub>p</sub>l<sub>e rea</sub>l<sub>-wor</sub>ld d<sub>oma</sub>i<sub>ns.</sub> Si<sub>nce wr</sub>it<sub>ers</sub> h<sub>ave</sub> di<sub>rec</sub>t <sub>access</sub> t<sub>o</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>con</sub>t<sub>en</sub>t<sub>, re</sub>li<sub>a</sub>bl<sub>e groun</sub>d<sub>-</sub>t<sub>ru</sub>th t<sub>ranscr</sub>i<sub>p</sub>ti<sub>ons can</sub> b<sub>e o</sub>bt<sub>a</sub>i<sub>ne</sub>d f<sub>rom</sub> th<sub>e source</sub> d<sub>ocumen</sub>t<sub>s.</sub>

Internet collection. Additional handwritten documents <sub>are co</sub>ll<sub>ec</sub>t<sub>e</sub>d f<sub>rom pu</sub>bli<sub>c</sub>l<sub>y ava</sub>il<sub>a</sub>bl<sub>e on</sub>li<sub>ne resources w</sub>h<sub>ere</sub> <sub>commun</sub>it<sub>y</sub> di<sub>scuss</sub>i<sub>ons</sub> <sub>or</sub> d<sub>ec</sub>i<sub>p</sub>h<sub>ermen</sub>t <sub>re</sub>f<sub>erences</sub> <sub>prov</sub>id<sub>e</sub> <sub>use</sub>f<sub>u</sub>l <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> f<sub>or anno</sub>t<sub>a</sub>ti<sub>on.</sub>

Aft<sub>er co</sub>ll<sub>ec</sub>ti<sub>on, a</sub>ll <sub>samp</sub>l<sub>es un</sub>d<sub>ergo au</sub>t<sub>oma</sub>ti<sub>c con</sub>t<sub>en</sub>t<sub>-</sub> <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> <sub>scann</sub>i<sub>ng</sub> t<sub>o</sub> <sub>remove</sub> d<sub>up</sub>li<sub>ca</sub>t<sub>es,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> <sub>manua</sub>l i<sub>nspec</sub>ti<sub>on o</sub>f <sub>am</sub>bi<sub>guous cases.</sub>

B<sub>enc</sub>h<sub>mar</sub>k <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> th<sub>en</sub> <sub>procee</sub>d<sub>s</sub> i<sub>n</sub> th<sub>ree</sub> <sub>s</sub>t<sub>ages.</sub>

Stage 1: MLLM-assisted pre-annotation. Three state-ofth<sub>e-ar</sub>t MLLM<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> t<sub>ranscr</sub>ib<sub>e every</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>.</sub> Th<sub>ese pre</sub>di<sub>c</sub>ti<sub>ons are no</sub>t <sub>use</sub>d <sub>as groun</sub>d t<sub>ru</sub>th<sub>,</sub> b<sub>u</sub>t <sub>serve</sub> <sub>as</sub> <sub>re</sub>f<sub>erence</sub> <sub>sugges</sub>ti<sub>ons</sub> f<sub>or</sub> h<sub>uman</sub> <sub>anno</sub>t<sub>a</sub>t<sub>ors,</sub> h<sub>e</sub>l<sub>p-</sub> i<sub>ng</sub> id<sub>en</sub>tif<sub>y</sub> <sub>am</sub>bi<sub>guous</sub> <sub>reg</sub>i<sub>ons</sub> <sub>an</sub>d i<sub>mprov</sub>i<sub>ng</sub> <sub>anno</sub>t<sub>a</sub>ti<sub>on</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency.</sub>

Stage 2: Human annotation. A primary annotator prod<sub>uces</sub> th<sub>e recogn</sub>iti<sub>on groun</sub>d t<sub>ru</sub>th f<sub>or every samp</sub>l<sub>e, repre-</sub> <sub>sen</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> M<sub>ar</sub>kd<sub>own</sub> f<sub>or</sub> f<sub>ree</sub> t<sub>ex</sub>t<sub>,</sub> LAT<sub>E</sub>X f<sub>or</sub> f<sub>ormu</sub>l<sub>as, an</sub>d HTML <table> for handwritten tables. Additionall<sub>y</sub>, each <sub>samp</sub>l<sub>e</sub> i<sub>s anno</sub>t<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith difi<sub>cu</sub>lt<sub>y ra</sub>ti<sub>ngs a</sub>l<sub>ong s</sub>i<sub>x</sub> di<sub>men-</sub> sions (visual de<sub>g</sub>radation, cursiveness, lin<sub>g</sub>uistic ambi<sub>g</sub>uit<sub>y</sub>, <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>comp</sub>l<sub>ex</sub>it<sub>y,</sub> d<sub>oma</sub>i<sub>n</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge, an</sub>d <sub>mu</sub>ltili<sub>ngua</sub> mixin<sub>g</sub>) to su<sub>pp</sub>ort future fine-<sub>g</sub>rained anal<sub>y</sub>sis.

D<sub>ur</sub>i<sub>ng anno</sub>t<sub>a</sub>ti<sub>on, anno</sub>t<sub>a</sub>t<sub>ors consu</sub>lt <sub>a</sub>ll <sub>ava</sub>il<sub>a</sub>bl<sub>e ev</sub>i<sub>-</sub> d<sub>ence,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> MLLM <sub>pre</sub>di<sub>c</sub>ti<sub>ons, wr</sub>it<sub>er-prov</sub>id<sub>e</sub>d t<sub>ran-</sub> <sub>scr</sub>i<sub>p</sub>t<sub>s,</sub> <sub>commun</sub>it<sub>y</sub> di<sub>scuss</sub>i<sub>ons,</sub> <sub>an</sub>d d<sub>oma</sub>i<sub>n</sub> <sub>exper</sub>t<sub>s</sub> <sub>w</sub>h<sub>en-</sub> <sub>ever necessary.</sub> Ch<sub>arac</sub>t<sub>ers</sub> th<sub>a</sub>t <sub>rema</sub>i<sub>n</sub> i<sub>mposs</sub>ibl<sub>e</sub> t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> <sub>a</sub>ft<sub>er</sub> <sub>ex</sub>h<sub>aus</sub>ti<sub>ve</sub> <sub>ver</sub>ifi<sub>ca</sub>ti<sub>on</sub> <sub>are</sub> <sub>mas</sub>k<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l i<sub>mage</sub> <sub>an</sub>d <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d f<sub>rom eva</sub>l<sub>ua</sub>ti<sub>on.</sub>

Stage 3: Independent verification. Every annotation is i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>rev</sub>i<sub>ewe</sub>d b<sub>y</sub> <sub>a</sub> <sub>secon</sub>d <sub>anno</sub>t<sub>a</sub>t<sub>or</sub> <sub>w</sub>h<sub>o</sub> <sub>per-</sub> f<sub>orms</sub> th<sub>e ver</sub>ifi<sub>ca</sub>ti<sub>on w</sub>ith<sub>ou</sub>t <sub>access</sub> t<sub>o</sub> th<sub>e pr</sub>i<sub>mary anno-</sub> t<sub>a</sub>t<sub>or</sub>’<sub>s reason</sub>i<sub>ng.</sub> Di<sub>sagreemen</sub>t<sub>s are reso</sub>l<sub>ve</sub>d th<sub>roug</sub>h di<sub>s-</sub> <sub>cuss</sub>i<sub>on</sub> <sub>un</sub>til <sub>consensus</sub> i<sub>s</sub> <sub>reac</sub>h<sub>e</sub>d<sub>.</sub> O<sub>n</sub>l<sub>y</sub> <sub>samp</sub>l<sub>es</sub> <sub>pass</sub>i<sub>ng</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>ver</sub>ifi<sub>ca</sub>ti<sub>on are</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> fi<sub>na</sub>l b<sub>enc</sub>h<sub>-</sub> <sub>mar</sub>k<sub>.</sub> Thi<sub>s mu</sub>lti<sub>-s</sub>t<sub>age qua</sub>lit<sub>y-con</sub>t<sub>ro</sub>l <sub>pro</sub>t<sub>oco</sub>l<sub>—au</sub>t<sub>oma</sub>ti<sub>c</sub> d<sub>e</sub>d<sub>up</sub>li<sub>ca</sub>ti<sub>on,</sub> MLLM<sub>-ass</sub>i<sub>s</sub>t<sub>e</sub>d <sub>anno</sub>t<sub>a</sub>ti<sub>on,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t d<sub>ou-</sub> ble review, and consensus-based adjudication—substantially <sub>re</sub>d<sub>uces anno</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>ncons</sub>i<sub>s</sub>t<sub>ency w</sub>hil<sub>e ensur</sub>i<sub>ng re</sub>li<sub>a</sub>bl<sub>e</sub> <sub>groun</sub>d t<sub>ru</sub>th f<sub>or su</sub>b<sub>sequen</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub>

## Evaluation Protocol

O<sub>ur eva</sub>l<sub>ua</sub>ti<sub>on pro</sub>t<sub>oco</sub>l <sub>cons</sub>i<sub>s</sub>t<sub>s o</sub>f th<sub>ree componen</sub>t<sub>s.</sub> Fi<sub>rs</sub>t<sub>,</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure-spec</sub>ifi<sub>c recogn</sub>iti<sub>on me</sub>t<sub>r</sub>i<sub>cs measure</sub> t<sub>ranscr</sub>i<sub>p</sub>ti<sub>on</sub> <sub>qua</sub>lit<sub>y across</sub> f<sub>ree</sub> t<sub>ex</sub>t<sub>,</sub> f<sub>ormu</sub>l<sub>as, an</sub>d t<sub>a</sub>bl<sub>es.</sub> S<sub>econ</sub>d<sub>,</sub> th<sub>e pro-</sub> posed Prior-Driven Error (PDE) metric quantifies whether <sub>recogn</sub>iti<sub>on</sub> f<sub>a</sub>il<sub>ures or</sub>i<sub>g</sub>i<sub>na</sub>t<sub>e</sub> f<sub>rom</sub> l<sub>anguage pr</sub>i<sub>ors ra</sub>th<sub>er</sub> th<sub>an v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence.</sub> Thi<sub>r</sub>d<sub>, ca</sub>lib<sub>ra</sub>t<sub>e</sub>d h<sub>uman</sub> b<sub>ase</sub>li<sub>nes pro-</sub> <sub>v</sub>id<sub>e a re</sub>f<sub>erence po</sub>i<sub>n</sub>t f<sub>or c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>z</sub>i<sub>ng</sub> h<sub>ow</sub> h<sub>uman an</sub>d <sub>mac</sub>hi<sub>ne errors</sub> dif<sub>er qua</sub>lit<sub>a</sub>ti<sub>ve</sub>l<sub>y.</sub>

## Recognition Evaluation

Followin<sub>g</sub> OmniDocBench (Ou<sub>y</sub>an<sub>g</sub> et al. 2025), difer-<sub>en</sub>t h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ures are eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure-spec</sub>ifi<sub>c</sub> <sub>me</sub>t<sub>r</sub>i<sub>cs.</sub>

• Free text. Character-level reco<sub>g</sub>nition <sub>q</sub>ualit<sub>y</sub> is measured usin<sub>g</sub> normalized Edit Distance (Edit↓) (Levenshtein 1966).

• Formulas. Formula reco<sub>g</sub>nition is evaluated usin<sub>g</sub> CDM (Character Detection Metric, CDM↑), which renders both <sub>pre</sub>di<sub>c</sub>ti<sub>on an</sub>d <sub>groun</sub>d t<sub>ru</sub>th b<sub>e</sub>f<sub>ore ma</sub>t<sub>c</sub>hi<sub>ng c</sub>h<sub>arac</sub>t<sub>er</sub> boundin<sub>g</sub> boxes (Wan<sub>g</sub> et al. 2025).

• Tables. Table understandin<sub>g</sub> is evaluated usin<sub>g</sub> TEDS (Tree Edit Distance Similarit<sub>y</sub>, TEDS↑), which jointl<sub>y</sub> <sub>measures s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>correc</sub>t<sub>ness an</sub>d t<sub>ex</sub>t<sub>ua</sub>l fid<sub>e</sub>lit<sub>y</sub> b<sub>ase</sub>d on HTML re<sub>p</sub>resentations (Zhon<sub>g</sub>, ShafieiBavani, and Jimeno Ye<sub>p</sub>es 2020).

![](images/a53b287fb240faf928302584b55697abcbd3a0a42fa07568c43da3239ffbfba0.jpg)  
Fi<sub>gure</sub> 1<sub>:</sub> O<sub>verv</sub>i<sub>ew</sub> <sub>o</sub>f th<sub>e</sub> W<sub>ild</sub>H<sub>and</sub>B<sub>ench</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> <sub>p</sub>i<sub>pe</sub>li<sub>ne.</sub>

T<sub>o</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>across</sub> dif<sub>eren</sub>t d<sub>ocumen</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ures, we repor</sub>t <sub>an overa</sub>ll <sub>score</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> O<sub>m-</sub> niDocBench (Ou<sub>y</sub>an<sub>g</sub> et al. 2025):

$$
\mathrm { O v e r a l l } = { \frac { ( 1 - \mathrm { E d i t } ) \times 1 0 0 + \mathrm { C D M } + \mathrm { T E D S } } { 3 } } .
$$

## Prior-Driven Error

R<sub>ecogn</sub>iti<sub>on accuracy a</sub>l<sub>one canno</sub>t di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h dif<sub>eren</sub>t t<sub>ypes o</sub>f <sub>recogn</sub>iti<sub>on</sub> f<sub>a</sub>il<sub>ures.</sub> T<sub>wo mo</sub>d<sub>e</sub>l<sub>s may ac</sub>hi<sub>eve</sub> id<sub>en-</sub> ti<sub>ca</sub>l <sub>recogn</sub>iti<sub>on</sub> <sub>accuracy</sub> <sub>w</sub>hil<sub>e</sub> <sub>ex</sub>hibiti<sub>ng</sub> <sub>very</sub> dif<sub>eren</sub>t b<sub>e-</sub> h<sub>av</sub>i<sub>ors: one may</sub> f<sub>a</sub>il b<sub>ecause</sub> it <sub>canno</sub>t <sub>correc</sub>tl<sub>y perce</sub>i<sub>ve</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> <sub>con</sub>t<sub>en</sub>t<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> th<sub>e</sub> <sub>o</sub>th<sub>er</sub> <sub>may</sub> <sub>genera</sub>t<sub>e</sub> <sub>v</sub>i<sub>su-</sub> <sub>a</sub>ll<sub>y</sub> <sub>unsuppor</sub>t<sub>e</sub>d <sub>ye</sub>t li<sub>ngu</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>p</sub>l<sub>aus</sub>ibl<sub>e</sub> t<sub>ranscr</sub>i<sub>p</sub>ti<sub>ons</sub> d<sub>ue</sub> t<sub>o</sub> <sub>excess</sub>i<sub>ve</sub> <sub>re</sub>li<sub>ance</sub> <sub>on</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>ors.</sub> Si<sub>nce</sub> th<sub>ese</sub> t<sub>wo</sub> f<sub>a</sub>il<sub>ure</sub> <sub>mo</sub>d<sub>es</sub> <sub>requ</sub>i<sub>re</sub> dif<sub>eren</sub>t <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> i<sub>mprovemen</sub>t<sub>s,</sub> W<sub>ild</sub>H<sub>and</sub>B<sub>ench</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> <sub>quan</sub>tifi<sub>es</sub> <sub>pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven</sub> <sub>recogn</sub>i<sub>-</sub> tion errors through the proposed Prior-Driven Error (PDE) <sub>me</sub>t<sub>r</sub>i<sub>c.</sub>

F<sub>or every pre</sub>di<sub>c</sub>ti<sub>on, mo</sub>d<sub>e</sub>l <sub>ou</sub>t<sub>pu</sub>t i<sub>s a</sub>li<sub>gne</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>correspon</sub>di<sub>ng groun</sub>d t<sub>ru</sub>th <sub>us</sub>i<sub>ng c</sub>h<sub>arac</sub>t<sub>er-</sub>l<sub>eve</sub>l <sub>a</sub>li<sub>gnmen</sub>t f<sub>or</sub> f<sub>ree</sub> t<sub>ex</sub>t <sub>an</sub>d f<sub>ormu</sub>l<sub>as, an</sub>d <sub>ce</sub>ll<sub>-</sub>l<sub>eve</sub>l <sub>a</sub>li<sub>gnmen</sub>t f<sub>or</sub> t<sub>a</sub>bl<sub>es.</sub> O<sub>n</sub>l<sub>y segmen</sub>t<sub>s w</sub>h<sub>ere pre</sub>di<sub>c</sub>ti<sub>on</sub> dif<sub>ers</sub> f<sub>rom</sub> th<sub>e groun</sub>d t<sub>ru</sub>th <sub>are</sub> <sub>cons</sub>id<sub>ere</sub>d<sub>.</sub> E<sub>ac</sub>h <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>segmen</sub>t i<sub>s</sub> th<sub>en</sub> <sub>score</sub>d usin<sub>g</sub> an external reference lan<sub>g</sub>ua<sub>g</sub>e model (Qwen3-8B-Base (Qwen Team 2025)). If the model-<sub>g</sub>enerated se<sub>g</sub>ment h<sub>as</sub> l<sub>ower</sub> <sub>perp</sub>l<sub>ex</sub>it<sub>y</sub> th<sub>an</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th segment, the error is regarded as prior-driven, indicating th<sub>a</sub>t th<sub>e mo</sub>d<sub>e</sub>l <sub>rep</sub>l<sub>aces v</sub>i<sub>sua</sub>ll<sub>y correc</sub>t b<sub>u</sub>t l<sub>ess common</sub> <sub>con</sub>t<sub>en</sub>t <sub>w</sub>ith <sub>a</sub> li<sub>ngu</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>more</sub> <sub>pro</sub>b<sub>a</sub>bl<sub>e</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ve.</sub> I<sub>n-</sub> <sub>ser</sub>t<sub>e</sub>d <sub>con</sub>t<sub>en</sub>t th<sub>a</sub>t h<sub>as</sub> <sub>no</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>groun</sub>d t<sub>ru</sub>th i<sub>s</sub> <sub>a</sub>l<sub>ways regar</sub>d<sub>e</sub>d <sub>as pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven s</sub>i<sub>nce</sub> it i<sub>s genera</sub>t<sub>e</sub>d <sub>so</sub>l<sub>e</sub>l<sub>y</sub> f<sub>rom</sub> i<sub>n</sub>t<sub>erna</sub>l l<sub>anguage</sub> <sub>pr</sub>i<sub>ors.</sub>

Formally, for a sample s, the Prior-Driven Error rate is com<sub>p</sub>ute<sup>d</sup> as

$$
\mathrm { P D E } ( s ) = \frac { \sum _ { i } \mathcal { W } [ \mathtt { p p l } ( m _ { i } ) < \mathtt { p p l } ( g _ { i } ) ] | m _ { i } | } { \sum _ { i } | m _ { i } | } ,
$$

<sub>w</sub>h<sub>ere</sub> $m _ { i }$ <sub>an</sub>d $g _ { i }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>pre</sub>di<sub>c</sub>ti<sub>on an</sub>d th<sub>e cor-</sub> <sub>respon</sub>di<sub>ng groun</sub>d<sub>-</sub>t<sub>ru</sub>th <sub>segmen</sub>t<sub>, respec</sub>ti<sub>ve</sub>l<sub>y.</sub> F<sub>or</sub> i<sub>nser</sub>t<sub>e</sub>d content<sub>,</sub> $\mathrm { p p l } ( g _ { i } ) = \infty$ <sub>.</sub> Th<sub>e</sub> <sub>summa</sub>ti<sub>on</sub> i<sub>s</sub> <sub>per</sub>f<sub>orme</sub>d <sub>on</sub>l<sub>y</sub> <sub>ove</sub>r mi<sub>s</sub>m<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d <sub>seg</sub>m<sub>e</sub>nt<sub>s.</sub> C<sub>o</sub>n<sub>seque</sub>ntl<sub>y,</sub> PDE m<sub>easu</sub>r<sub>es</sub> th<sub>e</sub> <sub>propor</sub>ti<sub>on</sub> <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l <sub>errors</sub> <sub>a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>a</sub>bl<sub>e</sub> t<sub>o</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>-</sub> <sub>ors ra</sub>th<sub>er</sub> th<sub>an v</sub>i<sub>sua</sub>l <sub>percep</sub>ti<sub>on,</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>re</sub>fl<sub>ec</sub>ti<sub>ng overa</sub>ll <sub>recogn</sub>iti<sub>on accuracy.</sub> Th<sub>e overa</sub>ll PDE <sub>score repor</sub>t<sub>e</sub>d f<sub>or</sub> <sub>eac</sub>h <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> th<sub>e ar</sub>ith<sub>me</sub>ti<sub>c mean o</sub>f th<sub>e per-ca</sub>t<sub>egory</sub> PDE rates (text, table, formula).

## Human Baselines

W<sub>ild</sub>H<sub>and</sub>B<sub>ench a</sub>dditi<sub>ona</sub>ll<sub>y es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es ca</sub>lib<sub>ra</sub>t<sub>e</sub>d h<sub>uman</sub> b<sub>ase</sub>li<sub>nes</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>same</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>pro</sub>t<sub>oco</sub>l<sub>.</sub> H<sub>uman</sub> <sub>par</sub>ti<sub>c</sub>i<sub>-</sub> <sub>pan</sub>t<sub>s</sub> <sub>rece</sub>i<sub>ve</sub> <sub>exac</sub>tl<sub>y</sub> th<sub>e</sub> <sub>same</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t i<sub>mages</sub> <sub>as</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>an</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>pro</sub>d<sub>uce</sub> <sub>recog-</sub> <sub>n</sub>iti<sub>on ou</sub>t<sub>pu</sub>t<sub>s us</sub>i<sub>ng</sub> th<sub>e same</sub> t<sub>arge</sub>t f<sub>orma</sub>t<sub>s.</sub> P<sub>er</sub>f<sub>ormance</sub> i<sub>s</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>me</sub>t<sub>r</sub>i<sub>cs</sub> d<sub>escr</sub>ib<sub>e</sub>d <sub>a</sub>b<sub>ove.</sub>

T<sub>o</sub> <sub>ensure</sub> <sub>a</sub> f<sub>a</sub>i<sub>r</sub> <sub>compar</sub>i<sub>son,</sub> h<sub>uman</sub> <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s</sub> <sub>per</sub>f<sub>orm</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e-pass</sub> <sub>recogn</sub>iti<sub>on</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>access</sub> t<sub>o</sub> <sub>any</sub> <sub>aux</sub>ili<sub>ary</sub> <sub>re-</sub> <sub>sources.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar,</sub> th<sub>ey are no</sub>t <sub>a</sub>ll<sub>owe</sub>d t<sub>o consu</sub>lt MLLM <sub>pre</sub>di<sub>c</sub>ti<sub>ons,</sub> <sub>wr</sub>it<sub>er-prov</sub>id<sub>e</sub>d t<sub>ranscr</sub>i<sub>p</sub>t<sub>s,</sub> <sub>commun</sub>it<sub>y</sub> di<sub>scus-</sub> <sub>s</sub>i<sub>ons,</sub> <sub>or</sub> d<sub>oma</sub>i<sub>n</sub> <sub>exper</sub>t<sub>s.</sub> Thi<sub>s</sub> <sub>se</sub>tti<sub>ng</sub> dif<sub>ers</sub> i<sub>n</sub>t<sub>en</sub>ti<sub>ona</sub>ll<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>cons</sub>t<sub>ruc</sub>ti<sub>on process, w</sub>h<sub>ere mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>ev</sub>id<sub>ence sources are</sub> i<sub>n</sub>t<sub>egra</sub>t<sub>e</sub>d t<sub>o es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>re</sub>li<sub>a</sub>bl<sub>e groun</sub>d t<sub>ru</sub>th th<sub>roug</sub>h <sub>consensus-</sub>b<sub>ase</sub>d <sub>anno</sub>t<sub>a</sub>ti<sub>on.</sub>

U<sub>n</sub>d<sub>er</sub> thi<sub>s pro</sub>t<sub>oco</sub>l<sub>,</sub> h<sub>uman par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s ac</sub>hi<sub>eve an overa</sub>ll score of 77.09%, which is hi<sub>g</sub>her than the best MLLM (Gemini 3.1 Pro, 71.85%). Detailed <sub>p</sub>er-cate<sub>g</sub>or<sub>y</sub> com<sub>p</sub>arisons are <sub>presen</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> E<sub>xper</sub>i<sub>men</sub>t<sub>s</sub> <sub>sec</sub>ti<sub>on.</sub>

## Experiments

## Evaluated Models

W<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub> 18 <sub>mo</sub>d<sub>e</sub>l<sub>s groupe</sub>d i<sub>n</sub>t<sub>o</sub> th<sub>ree ca</sub>t<sub>egor</sub>i<sub>es:</sub> <sub>p</sub>ro<sub>p</sub>rietar<sub>y</sub> VLMs (Claude O<sub>p</sub>us 4.8 (Anthro<sub>p</sub>ic 2026), Gemini 3.1 Pro (Goo<sub>g</sub>le Dee<sub>p</sub>Mind 2026a), Gemini 3.5 Flash (Goo<sub>g</sub>le Dee<sub>p</sub>Mind 2026b)), o<sub>p</sub>en-source <sub>g</sub>eneral VLMs (InternVL3.5 (O<sub>p</sub>enGVLab 2025), Qwen3-VL (Bai et al. 2025), Qwen3.5-Plus (Alibaba Cloud 2026), Kimi-K3 (Kimi Team 2026)), and o<sub>p</sub>en-source OCR-focused VLMs (Dee<sub>p</sub>Seek-OCR-V2 (Wei, Sun, and Li 2026), MinerU2.5 (Niu et al. 2025), MinerU2.5-Pro (Wan<sub>g</sub> et al. 2026), Unlimited-OCR (Yin et al. 2026), Dots.ocr (Li et al. 2025), Dots.mocr (Zhen<sub>g</sub> et al. 2026), Hun<sub>y</sub>uanOCR (Hun-<sub>y</sub>uan Vision Team et al. 2025), Qianfan-OCR (Don<sub>g</sub> et al. 2026), FireRed-OCR (Wu et al. 2026), GLM-OCR (Duan et al. 2026), PaddleOCR-VL-1.6 (Zhan<sub>g</sub> et al. 2026)).

All <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>are</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>an</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>pos</sub>t<sub>-process</sub>i<sub>ng</sub> <sub>p</sub>i<sub>pe</sub>li<sub>ne.</sub> G<sub>enera</sub>l VLM<sub>s use</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d O<sub>mn</sub>iD<sub>oc</sub>B<sub>enc</sub>h <sub>p</sub>rom<sub>p</sub>t tem<sub>p</sub>late (Ou<sub>y</sub>an<sub>g</sub> et al. 2025), while OCR-focused <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>use</sub> th<sub>e</sub>i<sub>r</sub> <sub>respec</sub>ti<sub>ve</sub> <sub>recommen</sub>d<sub>e</sub>d <sub>promp</sub>t <sub>se</sub>tti<sub>ngs</sub> <sub>or</sub> API<sub>s.</sub>

M<sub>o</sub>d<sub>e</sub>l<sub>s are se</sub>l<sub>ec</sub>t<sub>e</sub>d t<sub>o represen</sub>t th<sub>e curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e o</sub>f th<sub>e</sub> <sub>ar</sub>t i<sub>n</sub> d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng, w</sub>ith <sub>re</sub>f<sub>erence</sub> t<sub>o</sub> th<sub>e</sub> O<sub>m-</sub> niDocBench leaderboard (Ou<sub>y</sub>an<sub>g</sub> et al. 2025).

## Main Results

T<sub>a</sub>bl<sub>e</sub> 2 <sub>presen</sub>t<sub>s</sub> th<sub>e ma</sub>i<sub>n resu</sub>lt<sub>s.</sub> Th<sub>e mos</sub>t <sub>s</sub>t<sub>r</sub>iki<sub>ng</sub> fi<sub>n</sub>d<sub>-</sub> i<sub>ng</sub> i<sub>s</sub> th<sub>e per</sub>f<sub>ormance</sub> d<sub>egra</sub>d<sub>a</sub>ti<sub>on</sub> f<sub>rom pr</sub>i<sub>n</sub>t<sub>e</sub>d t<sub>o</sub> h<sub>an</sub>d<sub>-</sub> <sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>s: mo</sub>d<sub>e</sub>l<sub>s</sub> th<sub>a</sub>t <sub>excee</sub>d 90% <sub>on</sub> O<sub>m-</sub> niDocBench (Ou<sub>y</sub>an<sub>g</sub> et al. 2025) dro<sub>p</sub> to 71.85% at b<sub>es</sub>t <sub>on</sub> W<sub>ild</sub>H<sub>and</sub>B<sub>ench, con</sub>fi<sub>rm</sub>i<sub>ng a pers</sub>i<sub>s</sub>t<sub>en</sub>t “h<sub>an</sub>d<sub>-</sub> <sub>wr</sub>itt<sub>en gap.</sub>” F<sub>ur</sub>th<sub>ermore,</sub> OCR<sub>-</sub>f<sub>ocuse</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s—w</sub>hi<sub>c</sub>h <sub>rou</sub>ti<sub>ne</sub>l<sub>y ou</sub>t<sub>per</sub>f<sub>orm genera</sub>l VLM<sub>s on pr</sub>i<sub>n</sub>t<sub>e</sub>d<sub>-</sub>d<sub>ocumen</sub>t b<sub>enc</sub>h<sub>mar</sub>k<sub>s—</sub>f<sub>a</sub>ll b<sub>e</sub>hi<sub>n</sub>d <sub>genera</sub>l VLM<sub>s on w</sub>ild h<sub>an</sub>d<sub>wr</sub>it<sub>-</sub> i<sub>ng.</sub> Th<sub>e</sub> t<sub>op s</sub>i<sub>x mo</sub>d<sub>e</sub>l<sub>s on</sub> W<sub>ild</sub>H<sub>and</sub>B<sub>ench are a</sub>ll <sub>genera</sub>l<sub>-</sub> <sub>purpose, sugges</sub>ti<sub>ng</sub> th<sub>a</sub>t l<sub>arger mo</sub>d<sub>e</sub>l <sub>capac</sub>it<sub>y an</sub>d <sub>more</sub> di<sub>-</sub> <sub>verse</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a</sub> <sub>ena</sub>bl<sub>e</sub> <sub>s</sub>t<sub>ronger</sub> <sub>genera</sub>li<sub>za</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> hi<sub>g</sub>h <sub>var</sub>i<sub>a</sub>bilit<sub>y o</sub>f h<sub>an</sub>d<sub>wr</sub>itt<sub>en con</sub>t<sub>en</sub>t<sub>, w</sub>hil<sub>e</sub> OCR<sub>-</sub>f<sub>ocuse</sub>d <sub>mo</sub>d<sub>-</sub> <sub>e</sub>l<sub>s are spec</sub>i<sub>a</sub>li<sub>ze</sub>d f<sub>or</sub> th<sub>e regu</sub>l<sub>ar</sub>it<sub>y o</sub>f <sub>pr</sub>i<sub>n</sub>t<sub>e</sub>d l<sub>ayou</sub>t<sub>s.</sub>

T<sub>a</sub>bl<sub>es rema</sub>i<sub>n</sub> th<sub>e mos</sub>t <sub>c</sub>h<sub>a</sub>ll<sub>eng</sub>i<sub>ng ca</sub>t<sub>egory: even</sub> th<sub>e</sub> best TEDS score (60.44) indicates nearl<sub>y</sub> 40% structural <sub>an</sub>d <sub>con</sub>t<sub>en</sub>t <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h<sub>, an</sub>d t<sub>a</sub>bl<sub>e scores s</sub>h<sub>ow</sub> th<sub>e sma</sub>ll<sub>es</sub>t <sub>cross-mo</sub>d<sub>e</sub>l <sub>var</sub>i<sub>ance</sub> <sub>among</sub> f<sub>unc</sub>ti<sub>ona</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> T<sub>ex</sub>t <sub>recog-</sub> <sub>n</sub>iti<sub>on</sub> <sub>s</sub>h<sub>ows</sub> th<sub>e</sub> <sub>w</sub>id<sub>es</sub>t <sub>sprea</sub>d<sub>,</sub> <sub>re</sub>fl<sub>ec</sub>ti<sub>ng</sub> l<sub>arge</sub> dif<sub>erences</sub> i<sub>n</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er-</sub>l<sub>eve</sub>l fid<sub>e</sub>lit<sub>y.</sub> G<sub>em</sub>i<sub>n</sub>i 3<sub>.</sub>1 P<sub>ro</sub> l<sub>ea</sub>d<sub>s</sub> i<sub>n</sub> <sub>a</sub>ll th<sub>ree</sub> <sub>ca</sub>t<sub>egor</sub>i<sub>es an</sub>d <sub>overa</sub>ll<sub>,</sub> th<sub>oug</sub>h Ki<sub>m</sub>i<sub>-</sub>K3 i<sub>s compe</sub>titi<sub>ve on</sub> text edit distance (26.07 vs. 24.32) and Gemini 3.5 Flash on formula CDM (78.22 vs. 79.42).

## Human vs. Model Analysis

Th<sub>e</sub> h<sub>uman</sub> b<sub>ase</sub>li<sub>ne ac</sub>hi<sub>eves an overa</sub>ll <sub>score o</sub>f 77<sub>.</sub>09%<sub>—</sub> higher than the best MLLM (Gemini 3.1 Pro, 71.85%) by 5<sub>.</sub>24 <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e po</sub>i<sub>n</sub>t<sub>s.</sub> Whil<sub>e</sub> thi<sub>s con</sub>fi<sub>rms</sub> th<sub>a</sub>t h<sub>uman per-</sub> f<sub>ormance rema</sub>i<sub>ns</sub> th<sub>e upper</sub> b<sub>oun</sub>d<sub>,</sub> th<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y mo</sub>d<sub>es</sub>t <sub>gap</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> th<sub>a</sub>t <sub>w</sub>ild h<sub>an</sub>d<sub>wr</sub>iti<sub>ng</sub> <sub>c</sub>h<sub>a</sub>ll<sub>enges</sub> h<sub>umans</sub> <sub>an</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s a</sub>lik<sub>e.</sub>

Th<sub>e per-ca</sub>t<sub>egory</sub> b<sub>rea</sub>kd<sub>own revea</sub>l<sub>s w</sub>h<sub>ere mo</sub>d<sub>e</sub>l<sub>s s</sub>till l<sub>ag</sub> b<sub>e</sub>hi<sub>n</sub>d h<sub>umans an</sub>d <sub>w</sub>h<sub>ere</sub> th<sub>ey</sub> h<sub>ave caug</sub>ht <sub>up.</sub> O<sub>n</sub> text, the best model actuall<sub>y</sub> sur<sub>p</sub>asses humans (Edit: 30.08 vs. 24.32), indicatin<sub>g</sub> that MLLMs <sub>p</sub>roduce more format-<sub>comp</sub>li<sub>an</sub>t t<sub>ranscr</sub>i<sub>p</sub>ti<sub>ons</sub> <sub>even</sub> th<sub>oug</sub>h h<sub>umans</sub> <sub>may</sub> <sub>ac</sub>hi<sub>eve</sub> su<sub>p</sub>erior character-level reco<sub>g</sub>nition. On tables (TEDS: 74.78 vs. 60.44), humans substantiall<sub>y</sub> out<sub>p</sub>erform all models b<sub>y</sub> <sub>over</sub> 14 <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e po</sub>i<sub>n</sub>t<sub>s, re</sub>fl<sub>ec</sub>ti<sub>ng super</sub>i<sub>or s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>un</sub>d<sub>er-</sub> standin<sub>g</sub> of irre<sub>g</sub>ular handwritten tables. On formulas (CDM: 86.56 vs. 79.42), humans also lead, thou<sub>g</sub>h the <sub>g</sub>a<sub>p</sub> is smaller (7.14 <sub>p</sub>oints), indicatin<sub>g</sub> that formula reco<sub>g</sub>nition in current MLLM<sub>s</sub> i<sub>s re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y ma</sub>t<sub>ure.</sub>

C<sub>ruc</sub>i<sub>a</sub>ll<sub>y,</sub> h<sub>uman errors are qua</sub>lit<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> dif<sub>eren</sub>t f<sub>rom</sub> <sub>mo</sub>d<sub>e</sub>l <sub>errors.</sub> Th<sub>e</sub> PDE <sub>ana</sub>l<sub>ys</sub>i<sub>s s</sub>h<sub>ows</sub> th<sub>a</sub>t h<sub>uman</sub> PDE i<sub>s</sub> 49.27%, substantiall<sub>y</sub> lower than all evaluated models (minimum 63.28%), indicatin<sub>g</sub> that human errors are more evenl<sub>y</sub> <sub>sp</sub>lit b<sub>e</sub>t<sub>ween</sub> <sub>v</sub>i<sub>sua</sub>l <sub>m</sub>i<sub>srecogn</sub>iti<sub>on</sub> <sub>an</sub>d <sub>pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven</sub> <sub>su</sub>b<sub>s</sub>ti<sub>-</sub> t<sub>u</sub>ti<sub>on.</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s,</sub> i<sub>n</sub> <sub>con</sub>t<sub>ras</sub>t<sub>,</sub> <sub>pro</sub>d<sub>uce</sub> di<sub>spropor</sub>ti<sub>ona</sub>t<sub>e</sub>l<sub>y</sub> <sub>pr</sub>i<sub>or-</sub> d<sub>r</sub>i<sub>ven</sub> <sub>errors—genera</sub>ti<sub>ng</sub> fl<sub>uen</sub>t b<sub>u</sub>t <sub>v</sub>i<sub>sua</sub>ll<sub>y</sub> <sub>unsuppor</sub>t<sub>e</sub>d <sub>ou</sub>t<sub>pu</sub>t <sub>w</sub>h<sub>ere</sub> h<sub>umans</sub> t<sub>en</sub>d t<sub>owar</sub>d <sub>conserva</sub>ti<sub>ve</sub> <sub>par</sub>ti<sub>a</sub>l t<sub>ran-</sub> <sub>scr</sub>i<sub>p</sub>ti<sub>ons.</sub> Thi<sub>s</sub> <sub>qua</sub>lit<sub>a</sub>ti<sub>ve</sub> <sub>gap</sub> i<sub>n</sub> f<sub>a</sub>il<sub>ure</sub> <sub>mo</sub>d<sub>es</sub> <sub>pers</sub>i<sub>s</sub>t<sub>s</sub> d<sub>e-</sub> <sub>sp</sub>it<sub>e</sub> th<sub>e</sub> <sub>narrow</sub>i<sub>ng</sub> <sub>quan</sub>tit<sub>a</sub>ti<sub>ve</sub> <sub>gap</sub> i<sub>n</sub> <sub>overa</sub>ll <sub>accuracy.</sub>

## Error Pattern Analysis

T<sub>a</sub>bl<sub>e</sub> 2 <sub>a</sub>l<sub>so repor</sub>t<sub>s</sub> th<sub>e pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven error ra</sub>t<sub>e</sub> d<sub>ecompose</sub>d b<sub>y</sub> <sub>con</sub>t<sub>en</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ure;</sub> Fi<sub>gure</sub> 2 <sub>prov</sub>id<sub>es qua</sub>lit<sub>a</sub>ti<sub>ve examp</sub>l<sub>es o</sub>f t<sub>a</sub>bl<sub>e pars</sub>i<sub>ng errors across mo</sub>d<sub>e</sub>l t<sub>ypes an</sub>d th<sub>e</sub> h<sub>uman</sub> b<sub>ase-</sub> li<sub>ne.</sub> A<sub>mong mo</sub>d<sub>e</sub>l<sub>s w</sub>ith f<sub>unc</sub>ti<sub>ona</sub>l <sub>v</sub>i<sub>sua</sub>l <sub>enco</sub>d<sub>ers,</sub> PDE <sub>ra</sub>t<sub>es range</sub> f<sub>rom</sub> 63% t<sub>o</sub> 79% <sub>w</sub>ith<sub>ou</sub>t <sub>a s</sub>i<sub>mp</sub>l<sub>e corre</sub>l<sub>a</sub>ti<sub>on</sub> t<sub>o</sub> <sub>accu</sub>r<sub>acy.</sub> N<sub>o</sub>t<sub>a</sub>bl<sub>y,</sub> <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>l VLM<sub>s</sub> <sub>w</sub>ith <sub>s</sub>tr<sub>o</sub>n<sub>g</sub> l<sub>a</sub>n<sub>guage</sub> ca<sub>p</sub>abilities show relativel<sub>y</sub> hi<sub>g</sub>h rates (Gemini 3.5 Flash: 79.21%, Qwen3.5-Plus: 77.82%, Gemini 3.1 Pro: 77.51%), <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> l<sub>owes</sub>t PDE <sub>ra</sub>t<sub>es</sub> b<sub>e</sub>l<sub>ong</sub> t<sub>o cer</sub>t<sub>a</sub>i<sub>n compac</sub>t OCR<sub>-</sub> focused models (GLM-OCR: 63.28%, PaddleOCR-VL-1.6: 64.48%).

Thi<sub>s revea</sub>l<sub>s a nuance</sub>d <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>p</sub> b<sub>e</sub>t<sub>ween mo</sub>d<sub>e</sub>l <sub>capa-</sub> bilit<sub>y an</sub>d <sub>error</sub> t<sub>ype.</sub> At th<sub>e ex</sub>t<sub>reme, mo</sub>d<sub>e</sub>l<sub>s w</sub>ith <sub>severe</sub>l<sub>y</sub> li<sub>m</sub>it<sub>e</sub>d <sub>v</sub>i<sub>sua</sub>l d<sub>eco</sub>di<sub>ng pro</sub>d<sub>uce a</sub>l<sub>mos</sub>t <sub>exc</sub>l<sub>us</sub>i<sub>ve</sub>l<sub>y pr</sub>i<sub>or-</sub> d<sub>r</sub>i<sub>ven</sub> <sub>errors.</sub> A<sub>mong</sub> b<sub>e</sub>tt<sub>er-per</sub>f<sub>orm</sub>i<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> <sub>power</sub>f<sub>u</sub>l l<sub>anguage pr</sub>i<sub>ors ac</sub>t <sub>as a</sub> d<sub>ou</sub>bl<sub>e-e</sub>d<sub>ge</sub>d <sub>swor</sub>d<sub>:</sub> th<sub>ey</sub> b<sub>oos</sub>t <sub>accuracy</sub> b<sub>y correc</sub>tl<sub>y</sub> i<sub>n</sub>f<sub>err</sub>i<sub>ng am</sub>bi<sub>guous c</sub>h<sub>arac</sub>t<sub>ers,</sub> b<sub>u</sub>t <sub>w</sub>h<sub>en errors</sub> d<sub>o occur,</sub> th<sub>ose errors are</sub> di<sub>spropor</sub>ti<sub>ona</sub>t<sub>e</sub>l<sub>y</sub> <sub>pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven—</sub>th<sub>e mo</sub>d<sub>e</sub>l’<sub>s s</sub>t<sub>rong</sub> l<sub>anguage capa</sub>bilit<sub>y causes</sub> it t<sub>o con</sub>fid<sub>en</sub>tl<sub>y genera</sub>t<sub>e p</sub>l<sub>aus</sub>ibl<sub>e a</sub>lt<sub>erna</sub>ti<sub>ves ra</sub>th<sub>er</sub> th<sub>an</sub> f<sub>a</sub>il <sub>s</sub>il<sub>en</sub>tl<sub>y.</sub> C<sub>onverse</sub>l<sub>y,</sub> th<sub>e mo</sub>d<sub>e</sub>l<sub>s w</sub>ith th<sub>e</sub> l<sub>owes</sub>t PDE rates (GLM-OCR, PaddleOCR-VL) are com<sub>p</sub>act models that <sub>pro</sub>d<sub>uce a</sub> l<sub>arger propor</sub>ti<sub>on o</sub>f <sub>v</sub>i<sub>sua</sub>l <sub>m</sub>i<sub>srecogn</sub>iti<sub>on errors</sub> (e.<sub>g</sub>., confusin<sub>g</sub> similar-lookin<sub>g</sub> characters) rather than <sub>g</sub>en-<sub>era</sub>ti<sub>ng</sub> fl<sub>uen</sub>t b<sub>u</sub>t <sub>unsuppor</sub>t<sub>e</sub>d t<sub>ex</sub>t<sub>.</sub>

Th<sub>e per-s</sub>t<sub>ruc</sub>t<sub>ure</sub> b<sub>rea</sub>kd<sub>own revea</sub>l<sub>s</sub> th<sub>a</sub>t f<sub>ormu</sub>l<sub>a errors</sub> are almost entirel<sub>y p</sub>rior-driven across all models (87–98%), lik<sub>e</sub>l<sub>y</sub> b<sub>ecause</sub> L<sub>a</sub>T<sub>e</sub>X <sub>syn</sub>t<sub>ax</sub> i<sub>s</sub> i<sub>n</sub>h<sub>eren</sub>tl<sub>y</sub> l<sub>ow-perp</sub>l<sub>ex</sub>it<sub>y—</sub> <sub>even</sub> l<sub>eg</sub>iti<sub>ma</sub>t<sub>e</sub> f<sub>orma</sub>t dif<sub>erences sa</sub>ti<sub>s</sub>f<sub>y</sub> th<sub>e pp</sub>l <sub>cr</sub>it<sub>er</sub>i<sub>on.</sub> Text errors show the widest s<sub>p</sub>read (52–84%), reflectin<sub>g</sub> the <sub>grea</sub>t<sub>es</sub>t <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n mo</sub>d<sub>e</sub>l <sub>re</sub>li<sub>ance on</sub> l<sub>anguage pr</sub>i<sub>ors.</sub> T<sub>a</sub>bl<sub>e</sub> errors are moderate (42–90%), with weaker models showin<sub>g</sub> hi<sub>g</sub>h<sub>er</sub> <sub>ra</sub>t<sub>es.</sub>

![](images/a0b00038e434e4a4aebd8939ee3e998dd64b2fbdfed94042853029f9d704cec7.jpg)

Fi<sub>g</sub>ure 2: Qualitative comparison of handwritten table parsin<sub>g</sub> across <sub>g</sub>eneral-purpose models, OCR-specialized models, and the human baseline. The ri<sub>g</sub>ht <sub>p</sub>anel illustrates three error t<sub>yp</sub>es: table structure errors (E1), table content errors (E2), and <sub>p</sub>rior-driven errors (E3).
<table><tr><td></td><td></td><td></td><td colspan="4">Performance Evaluation (%)</td><td colspan="4">Prior-Driven Error (%)</td></tr><tr><td>Type</td><td>Model</td><td>Params</td><td>Text Edit↓</td><td>Table TEDS↑</td><td>Formula CDM↑</td><td>Overall ↑</td><td>Text↓</td><td>Table↓</td><td>Formula↓</td><td>Overall.↓</td></tr><tr><td rowspan="3">Proprietary</td><td>Claude Opus 4.8</td><td></td><td>53.30</td><td>52.27</td><td>74.11</td><td>57.70</td><td>72.78</td><td>59.32</td><td>91.66</td><td>74.59</td></tr><tr><td>Gemini 3.5 Flash</td><td></td><td>27.10</td><td>53.63</td><td>78.22</td><td>68.25</td><td>71.36</td><td>71.91</td><td>94.36</td><td>79.21</td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td>24.32</td><td>60.44</td><td>79.42</td><td>71.85</td><td>71.19</td><td>66.56</td><td>94.77</td><td>77.51</td></tr><tr><td rowspan="4">Open General</td><td>InternVL3.5</td><td>241B-A28B</td><td>46.80</td><td>41.22</td><td>64.00</td><td>52.80</td><td>72.62</td><td>66.38</td><td>90.03</td><td>76.34</td></tr><tr><td>女 Qwen3-VL</td><td>235B-A22B</td><td>34.30</td><td>51.81</td><td>69.06</td><td>62.19</td><td>70.01</td><td>62.18</td><td>91.33</td><td>74.51</td></tr><tr><td>Qwen3.5-Plus</td><td>397B-A17B</td><td>34.80</td><td>57.68</td><td>73.77</td><td>65.55</td><td>71.74</td><td>70.00</td><td>91.71</td><td>77.82</td></tr><tr><td>KKimi-K3</td><td>2.8T-104B</td><td>26.07</td><td>58.99</td><td>70.50</td><td>67.81</td><td>65.29</td><td>60.32</td><td>90.21</td><td>71.94</td></tr><tr><td rowspan="9">Open OCR</td><td>DeepSeek-OCR2</td><td>3B-A0.5B</td><td>91.70</td><td>1.77</td><td>18.33</td><td>9.46</td><td>84.06</td><td>90.00</td><td>98.35</td><td>90.80</td></tr><tr><td> MinerU-2.5</td><td>1.2B</td><td>83.80</td><td>41.19</td><td>36.56</td><td>31.33</td><td>73.80</td><td>55.21</td><td>92.16</td><td>73.72</td></tr><tr><td>Unlimited-OCR</td><td>3B-A0.5B</td><td>64.90</td><td>34.41</td><td>51.62</td><td>40.38</td><td>61.67</td><td>55.16</td><td>89.47</td><td>68.77</td></tr><tr><td>R Dots.ocr</td><td>3B</td><td>56.68</td><td>34.70</td><td>53.17</td><td>43.73</td><td>70.29</td><td>59.75</td><td>90.98</td><td>73.67</td></tr><tr><td>R Dots.mocr</td><td>3B</td><td>49.54</td><td>36.60</td><td>62.85</td><td>49.97</td><td>63.23</td><td>59.56</td><td>89.73</td><td>70.84</td></tr><tr><td>HunyuanOCR</td><td>1.0B</td><td>38.50</td><td>47.02</td><td>45.81</td><td>51.44</td><td>67.13</td><td>55.55</td><td>92.08</td><td>71.59</td></tr><tr><td> MinerU2.5-Pro</td><td>1.2B</td><td>59.80</td><td>53.16</td><td>61.79</td><td>51.73</td><td>61.23</td><td>47.46</td><td>92.80</td><td>67.16</td></tr><tr><td>品 Qianfan-OCR</td><td>4B</td><td>50.40</td><td>45.22</td><td>61.40</td><td>52.07</td><td>64.07</td><td>61.29</td><td>94.04</td><td>73.13</td></tr><tr><td>R FireRed-OCR</td><td>2B</td><td>49.00</td><td>44.30 57.38</td><td>63.11</td><td>52.82 54.06</td><td>64.70</td><td>64.25</td><td>93.10</td><td>74.02</td></tr><tr><td>C GLM-OCR</td><td></td><td>0.9B</td><td>54.20</td><td>59.01</td><td></td><td></td><td>57.02</td><td>41.92</td><td>90.89</td><td>63.28</td></tr><tr><td></td><td>PaddleOCR-VL-1.6</td><td>0.9B</td><td>41.52</td><td>52.41</td><td>51.75</td><td>54.21</td><td>52.59</td><td>53.01</td><td>87.83</td><td>64.48</td></tr><tr><td>Human</td><td>Human</td><td>一</td><td>30.08</td><td>74.78</td><td>86.56</td><td>77.09</td><td>66.96</td><td>34.48</td><td>46.36</td><td>49.27</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> R<sub>ecogn</sub>iti<sub>on</sub> <sub>per</sub>f<sub>ormance</sub> <sub>an</sub>d <sub>pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven</sub> <sub>error</sub> f<sub>or</sub> <sub>a</sub>ll <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>an</sub>d th<sub>e</sub> h<sub>uman</sub> b<sub>ase</sub>li<sub>ne.</sub>

W<sub>e no</sub>t<sub>e</sub> li<sub>m</sub>it<sub>a</sub>ti<sub>ons o</sub>f thi<sub>s me</sub>t<sub>r</sub>i<sub>c:</sub> it <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>zes error</sub> type, not error quantity—a high prior-driven rate does not <sub>mean a mo</sub>d<sub>e</sub>l i<sub>s</sub> l<sub>ess re</sub>li<sub>a</sub>bl<sub>e, on</sub>l<sub>y</sub> th<sub>a</sub>t it<sub>s errors are more</sub> <sub>sys</sub>t<sub>ema</sub>ti<sub>c.</sub> Th<sub>e c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on</sub> d<sub>epen</sub>d<sub>s on a spec</sub>ifi<sub>c re</sub>f<sub>erence</sub> lan<sub>g</sub>ua<sub>g</sub>e model (Qwen3-8B-Base); a diferent model ma<sub>y</sub> <sub>s</sub>hift b<sub>oun</sub>d<sub>ary cases.</sub>

## Discussion

Th<sub>e</sub> fi<sub>n</sub>di<sub>ngs o</sub>f W<sub>ild</sub>H<sub>and</sub>B<sub>ench</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t h<sub>an</sub>d<sub>-</sub> <sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng rema</sub>i<sub>ns unso</sub>l<sub>ve</sub>d<sub>: mo</sub>d<sub>e</sub>l<sub>s</sub> exceedin<sub>g</sub> 90% on OmniDocBench (Ou<sub>y</sub>an<sub>g</sub> et al. 2025) dro<sub>p</sub> t<sub>o</sub> 71<sub>.</sub>85% <sub>a</sub>t b<sub>es</sub>t <sub>on</sub> W<sub>ild</sub>H<sub>and</sub>B<sub>ench, an</sub>d thi<sub>s</sub> d<sub>egra</sub>d<sub>a-</sub> ti<sub>on</sub> i<sub>s non-un</sub>if<sub>orm across</sub> d<sub>ocumen</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ures.</sub> M<sub>ore cr</sub>iti<sub>-</sub> <sub>ca</sub>ll<sub>y,</sub> th<sub>e</sub> PDE <sub>ana</sub>l<sub>ys</sub>i<sub>s revea</sub>l<sub>s</sub> th<sub>a</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d h<sub>uman errors</sub> <sub>are</sub> <sub>qua</sub>lit<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> dif<sub>eren</sub>t<sub>—</sub>63<sub>–</sub>79% <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l <sub>errors</sub> <sub>among</sub> f<sub>unc</sub>ti<sub>ona</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>t<sub>e</sub> f<sub>rom</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>ors,</sub> <sub>compare</sub>d t<sub>o</sub> 49% f<sub>or</sub> h<sub>umans.</sub> Thi<sub>s means</sub> th<sub>a</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>errors are sys</sub>t<sub>em-</sub> <sub>a</sub>ti<sub>ca</sub>ll<sub>y more</sub> d<sub>angerous:</sub> fl<sub>uen</sub>t<sub>, con</sub>fid<sub>en</sub>t<sub>, an</sub>d <sub>unsuppor</sub>t<sub>e</sub>d b<sub>y v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence.</sub>

Th<sub>ese</sub> fi<sub>n</sub>di<sub>ngs</sub> h<sub>ave</sub> i<sub>mp</sub>li<sub>ca</sub>ti<sub>ons</sub> f<sub>or sa</sub>f<sub>e</sub>t<sub>y-cr</sub>iti<sub>ca</sub>l <sub>app</sub>li<sub>-</sub> <sub>ca</sub>ti<sub>ons</sub> <sub>suc</sub>h <sub>as</sub> <sub>me</sub>di<sub>ca</sub>l <sub>recor</sub>d<sub>s,</sub> fi<sub>nanc</sub>i<sub>a</sub>l d<sub>ocumen</sub>t<sub>s,</sub> <sub>an</sub>d l<sub>ega</sub>l <sub>arc</sub>hi<sub>ves, w</sub>h<sub>ere pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven errors are par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y</sub> difi<sub>-</sub> cult to detect (Bai et al. 2024). Future models should o<sub>p</sub>timize <sub>no</sub>t <sub>on</sub>l<sub>y recogn</sub>iti<sub>on accuracy</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so v</sub>i<sub>sua</sub>l <sub>groun</sub>di<sub>ng an</sub>d <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y,</sub> l<sub>earn</sub>i<sub>ng</sub> t<sub>o</sub> <sub>express</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an con</sub>fid<sub>en</sub>tl<sub>y genera</sub>ti<sub>ng p</sub>l<sub>aus</sub>ibl<sub>e</sub> t<sub>ranscr</sub>i<sub>p</sub>ti<sub>ons w</sub>h<sub>en v</sub>i<sub>-</sub> <sub>sua</sub>l <sub>ev</sub>id<sub>ence</sub> i<sub>s</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>.</sub>

## Limitations

Benchmark scale and coverage. Although WildHand-B<sub>ench con</sub>t<sub>a</sub>i<sub>ns care</sub>f<sub>u</sub>ll<sub>y cura</sub>t<sub>e</sub>d h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> d<sub>ocumen</sub>t<sub>s</sub> <sub>w</sub>ith hi<sub>g</sub>h<sub>-qua</sub>lit<sub>y anno</sub>t<sub>a</sub>ti<sub>ons,</sub> it <sub>curren</sub>tl<sub>y cons</sub>i<sub>s</sub>t<sub>s o</sub>f 500 <sub>samp</sub>l<sub>es an</sub>d <sub>s</sub>h<sub>ou</sub>ld <sub>no</sub>t b<sub>e</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>e</sub>d <sub>as a popu</sub>l<sub>a</sub>ti<sub>on-</sub>l<sub>eve</sub>l <sub>es</sub>ti<sub>ma</sub>t<sub>e o</sub>f h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> OCR <sub>per</sub>f<sub>ormance.</sub> Th<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> small formula (52 ima<sub>g</sub>es) and table (81 ima<sub>g</sub>es) subsets li<sub>m</sub>it <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>power</sub> f<sub>or</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>hi<sub>ng</sub> hi<sub>g</sub>hl<sub>y compe</sub>titi<sub>ve</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> F<sub>ur</sub>th<sub>ermore,</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k f<sub>ocuses pr</sub>i<sub>mar</sub>il<sub>y on</sub> Chi<sub>nese an</sub>d E<sub>ng</sub>li<sub>s</sub>h h<sub>an</sub>d<sub>wr</sub>iti<sub>ng.</sub> F<sub>u</sub>t<sub>ure vers</sub>i<sub>ons w</sub>ill <sub>ex-</sub> <sub>pan</sub>d b<sub>o</sub>th l<sub>anguage coverage an</sub>d d<sub>a</sub>t<sub>ase</sub>t <sub>sca</sub>l<sub>e.</sub>

Annotation and human evaluation. Part of the bench-<sub>mar</sub>k i<sub>s co</sub>ll<sub>ec</sub>t<sub>e</sub>d f<sub>rom pu</sub>bli<sub>c</sub>l<sub>y ava</sub>il<sub>a</sub>bl<sub>e</sub> I<sub>n</sub>t<sub>erne</sub>t <sub>resources,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>may</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce source</sub> bi<sub>as</sub> d<sub>esp</sub>it<sub>e care</sub>f<sub>u</sub>l <sub>manua</sub>l <sub>screen</sub>i<sub>ng an</sub>d d<sub>e</sub>d<sub>up</sub>li<sub>ca</sub>ti<sub>on.</sub> C<sub>er</sub>t<sub>a</sub>i<sub>n</sub> h<sub>an</sub>d<sub>wr</sub>itt<sub>en scenar-</sub> i<sub>os, suc</sub>h <sub>as</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>l <sub>ca</sub>lli<sub>grap</sub>h<sub>y or severe</sub>l<sub>y</sub> d<sub>egra</sub>d<sub>e</sub>d manuscripts, inherently involve subjective interpretation, <sub>ma</sub>ki<sub>ng a un</sub>i<sub>versa</sub>ll<sub>y correc</sub>t t<sub>ranscr</sub>i<sub>p</sub>ti<sub>on</sub> difi<sub>cu</sub>lt t<sub>o</sub> d<sub>e-</sub> fi<sub>ne even un</sub>d<sub>er consensus-</sub>b<sub>ase</sub>d <sub>anno</sub>t<sub>a</sub>ti<sub>on.</sub> Th<sub>e repor</sub>t<sub>e</sub>d h<sub>uman</sub> b<sub>ase</sub>li<sub>ne s</sub>h<sub>ou</sub>ld th<sub>ere</sub>f<sub>ore</sub> b<sub>e regar</sub>d<sub>e</sub>d <sub>as a ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>re</sub>f<sub>erence ra</sub>th<sub>er</sub> th<sub>an an a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e upper</sub> b<sub>oun</sub>d<sub>.</sub> M<sub>oreover, a</sub>ll <sub>par</sub>ti<sub>c</sub>i<sub>pa</sub>ti<sub>ng anno</sub>t<sub>a</sub>t<sub>ors are na</sub>ti<sub>ve</sub> Chi<sub>nese spea</sub>k<sub>ers, w</sub>hi<sub>c</sub>h <sub>may un</sub>d<sub>eres</sub>ti<sub>ma</sub>t<sub>e ac</sub>hi<sub>eva</sub>bl<sub>e</sub> h<sub>uman per</sub>f<sub>ormance on</sub> hi<sub>g</sub>hl<sub>y</sub> <sub>curs</sub>i<sub>ve</sub> <sub>or</sub> <sub>s</sub>t<sub>y</sub>li<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> di<sub>verse</sub> E<sub>ng</sub>li<sub>s</sub>h h<sub>an</sub>d<sub>wr</sub>iti<sub>ng.</sub>

PDE metric. The PDE metric depends on an external reference lan<sub>g</sub>ua<sub>g</sub>e model (Qwen3-8B-Base (Qwen Team 2025)) for <sub>p</sub>er<sub>p</sub>lexit<sub>y</sub> estimation, and alternative reference <sub>mo</sub>d<sub>e</sub>l<sub>s may</sub> l<sub>ea</sub>d t<sub>o</sub> dif<sub>eren</sub>t <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>ons</sub> f<sub>or am</sub>bi<sub>guous</sub> b<sub>oun</sub>d<sub>ary cases.</sub> M<sub>oreover,</sub> PDE i<sub>s</sub> d<sub>es</sub>i<sub>gne</sub>d t<sub>o c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> error type rather than quantity—a higher PDE does not imply l<sub>ower mo</sub>d<sub>e</sub>l <sub>re</sub>li<sub>a</sub>bilit<sub>y, on</sub>l<sub>y</sub> th<sub>a</sub>t <sub>a</sub> l<sub>arger propor</sub>ti<sub>on o</sub>f <sub>errors</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>t<sub>e</sub> f<sub>rom</sub> l<sub>anguage</sub> <sub>pr</sub>i<sub>ors</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>v</sub>i<sub>sua</sub>l <sub>percep</sub>ti<sub>on.</sub>

## Conclusion

W<sub>e presen</sub>t<sub>e</sub>d W<sub>ild</sub>H<sub>and</sub>B<sub>ench, a</sub> b<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> h<sub>an</sub>d<sub>wr</sub>it<sub>-</sub> ten document understanding that jointly evaluates free text, t<sub>a</sub>bl<sub>es,</sub> <sub>an</sub>d f<sub>ormu</sub>l<sub>as</sub> <sub>w</sub>ith <sub>a</sub> <sub>un</sub>ifi<sub>e</sub>d <sub>pro</sub>t<sub>oco</sub>l <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> <sub>recog-</sub> <sub>n</sub>iti<sub>on me</sub>t<sub>r</sub>i<sub>cs,</sub> h<sub>uman</sub> b<sub>ase</sub>li<sub>nes, an</sub>d P<sub>r</sub>i<sub>or-</sub>D<sub>r</sub>i<sub>ven</sub> E<sub>rror ana</sub>l<sub>-</sub> <sub>ys</sub>i<sub>s.</sub> E<sub>xper</sub>i<sub>men</sub>t<sub>s</sub> <sub>on</sub> 18 <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>revea</sub>l <sub>a</sub> <sub>pers</sub>i<sub>s</sub>t<sub>en</sub>t h<sub>an</sub>d<sub>wr</sub>it<sub>-</sub> ten <sub>g</sub>a<sub>p</sub> (best model 71.85% vs. human 77.09%), with model <sub>errors qua</sub>lit<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> dif<sub>eren</sub>t f<sub>rom</sub> h<sub>uman errors—</sub>63<sub>–</sub>79% <sub>pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven</sub> f<sub>or mo</sub>d<sub>e</sub>l<sub>s vs.</sub> 49% f<sub>or</sub> h<sub>umans.</sub>

## References

Alibaba Cloud. 2026. Qwen3.5-Plus. Model Studio release d<sub>ocumen</sub>t<sub>a</sub>ti<sub>on.</sub> A<sub>ccesse</sub>d 2026<sub>-</sub>07<sub>-</sub>27<sub>.</sub>

A<sub>n</sub>th<sub>rop</sub>i<sub>c.</sub> 2026<sub>.</sub> I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>i<sub>ng</sub> Cl<sub>au</sub>d<sub>e</sub> O<sub>pus</sub> 4<sub>.</sub>8<sub>.</sub> Ofi<sub>c</sub>i<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>l <sub>re</sub>l<sub>ease.</sub> A<sub>ccesse</sub>d 2026<sub>-</sub>07<sub>-</sub>27<sub>.</sub>

B<sub>a</sub>i<sub>,</sub> S<sub>.;</sub> C<sub>a</sub>i<sub>,</sub> Y<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> R<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> K<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> X<sub>.;</sub> Ch<sub>e</sub>n<sub>g,</sub> Z<sub>.;</sub> Den<sub>g</sub>, L.; Din<sub>g</sub>, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

B<sub>a</sub>i<sub>,</sub> Z<sub>.;</sub> W<sub>ang,</sub> P<sub>.;</sub> Xi<sub>ao,</sub> T<sub>.;</sub> H<sub>e,</sub> T<sub>.;</sub> H<sub>an,</sub> Z<sub>.;</sub> Zh<sub>ang,</sub> Z<sub>.; an</sub>d Sh<sub>ou,</sub> M<sub>.</sub> Z<sub>.</sub> 2024<sub>.</sub> H<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>ti<sub>on o</sub>f M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l L<sub>arge</sub> L<sub>an-</sub> guage Models: A Survey. arXiv preprint arXiv:2404.18930.

D<sub>ong,</sub> D<sub>.;</sub> Zh<sub>eng,</sub> M<sub>.;</sub> X<sub>u,</sub> D<sub>.;</sub> L<sub>uo,</sub> C<sub>.;</sub> Zh<sub>uang,</sub> B<sub>.;</sub> Li<sub>,</sub> Y<sub>.;</sub> H<sub>e,</sub> R.; Wan<sub>g</sub>, H.; Zhan<sub>g</sub>, W.; Wan<sub>g</sub>, W.; et al. 2026. Qianfan-OCR<sub>:</sub> A U<sub>n</sub>ifi<sub>e</sub>d E<sub>n</sub>d<sub>-</sub>t<sub>o-</sub>E<sub>n</sub>d M<sub>o</sub>d<sub>e</sub>l f<sub>or</sub> D<sub>ocumen</sub>t I<sub>n</sub>t<sub>e</sub>lli<sub>-</sub> gence. arXiv preprint arXiv:2603.13398.

D<sub>uan,</sub> S<sub>.;</sub> X<sub>ue,</sub> Y<sub>.;</sub> W<sub>ang,</sub> W<sub>.;</sub> S<sub>u,</sub> Z<sub>.;</sub> Li<sub>u,</sub> H<sub>.;</sub> Y<sub>ang,</sub> S<sub>.;</sub> G<sub>an,</sub>G<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> G<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> Z<sub>.;</sub> Y<sub>a</sub>n<sub>,</sub> S<sub>.; e</sub>t <sub>a</sub>l<sub>.</sub> 2026<sub>.</sub> GLM<sub>-</sub>OCRTechnical Report. arXiv preprint arXiv:2603.10910.

Fu, L.; Yan<sub>g</sub>, B.; Kuan<sub>g</sub>, Z.; Son<sub>g</sub>, J.; Li, Y.; Zhu, L.; Luo, Q.; W<sub>ang,</sub> X<sub>.;</sub> L<sub>u,</sub> H<sub>.;</sub> H<sub>uang,</sub> M<sub>.;</sub> Li<sub>,</sub> Z<sub>.;</sub> T<sub>ang,</sub> G<sub>.;</sub> Sh<sub>an,</sub> B<sub>.;</sub> Li<sub>n,</sub> C.; Liu, Q.; Wu, B.; Fen<sub>g</sub>, H.; Liu, H.; Huan<sub>g</sub>, C.; Tan<sub>g</sub>, J.; Ch<sub>en,</sub> W<sub>.;</sub> Ji<sub>n,</sub> L<sub>.;</sub> Li<sub>u,</sub> Y<sub>.; an</sub>d B<sub>a</sub>i<sub>,</sub> X<sub>.</sub> 2025<sub>.</sub> OCRB<sub>enc</sub>h <sub>v</sub>2<sub>:</sub> A<sub>n</sub> I<sub>mprove</sub>d B<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> L<sub>arge</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l Models on Visual Text Localization and Reasoning. arXiv preprint arXiv:2501.00321.

G<sub>erva</sub>i<sub>s,</sub> P<sub>.;</sub> F<sub>a</sub>d<sub>eeva,</sub> A<sub>.; an</sub>d M<sub>a</sub>k<sub>sa</sub>i<sub>,</sub> A<sub>.</sub> 2024<sub>.</sub> M<sub>a</sub>thW<sub>r</sub>it<sub>-</sub> i<sub>ng:</sub> A D<sub>a</sub>t<sub>ase</sub>t f<sub>or</sub> H<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> M<sub>a</sub>th<sub>ema</sub>ti<sub>ca</sub>l E<sub>xpress</sub>i<sub>on</sub> Recognition. arXiv preprint arXiv:2404.10690.

Göb<sub>e</sub>l<sub>,</sub> M<sub>.;</sub> H<sub>assa</sub>n<sub>,</sub> T<sub>.;</sub> Or<sub>o,</sub> E<sub>.; a</sub>nd Or<sub>s</sub>i<sub>,</sub> G<sub>.</sub> 2013<sub>.</sub> ICDAR 2013 Table Competition. In 2013 12th International Conference on Document Analysis and Recognition, 1449–1453.

G<sub>oog</sub>l<sub>e</sub> D<sub>eep</sub>Mi<sub>n</sub>d<sub>.</sub> 2026<sub>a.</sub> G<sub>em</sub>i<sub>n</sub>i 3<sub>.</sub>1 P<sub>ro:</sub> A S<sub>mar</sub>t<sub>er</sub> M<sub>o</sub>d<sub>e</sub>l f<sub>or</sub> Y<sub>our</sub> M<sub>os</sub>t C<sub>omp</sub>l<sub>ex</sub> T<sub>as</sub>k<sub>s.</sub> Ofi<sub>c</sub>i<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>l <sub>re</sub>l<sub>ease.</sub> A<sub>c-</sub> <sub>cesse</sub>d 2026<sub>-</sub>07<sub>-</sub>27<sub>.</sub>

G<sub>oog</sub>l<sub>e</sub> D<sub>eep</sub>Mi<sub>n</sub>d<sub>.</sub> 2026b<sub>.</sub> G<sub>em</sub>i<sub>n</sub>i 3<sub>.</sub>5 Fl<sub>as</sub>h M<sub>o</sub>d<sub>e</sub>l C<sub>ar</sub>d<sub>.</sub> Ofi<sub>c</sub>i<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>l <sub>car</sub>d<sub>.</sub> A<sub>ccesse</sub>d 2026<sub>-</sub>07<sub>-</sub>27<sub>.</sub>

Gr<sub>os</sub>i<sub>c</sub>ki<sub>,</sub> E<sub>.;</sub> C<sub>a</sub>rré<sub>,</sub> M<sub>.;</sub> Br<sub>o</sub>din<sub>,</sub> J<sub>.-</sub>M<sub>.; a</sub>nd G<sub>eo</sub>fr<sub>o</sub>i<sub>s,</sub> E<sub>.</sub> 2009<sub>.</sub> R<sub>esu</sub>lt<sub>s o</sub>f th<sub>e</sub> RIMES E<sub>va</sub>l<sub>ua</sub>ti<sub>o</sub>n C<sub>a</sub>m<sub>pa</sub>i<sub>g</sub>n f<sub>o</sub>r H<sub>a</sub>nd<sub>-</sub> written Mail Processing. In 2009 10th International Conference on Document Analysis and Recognition, 941–945.

G<sub>uan,</sub> T<sub>.;</sub> Li<sub>u,</sub> F<sub>.;</sub> W<sub>u,</sub> X<sub>.;</sub> Xi<sub>an,</sub> R<sub>.;</sub> Li<sub>,</sub> Z<sub>.;</sub> Li<sub>u,</sub> X<sub>.;</sub> W<sub>ang,</sub> X<sub>.;</sub> Ch<sub>en,</sub> L<sub>.;</sub> H<sub>uang,</sub> F<sub>.;</sub> Y<sub>acoo</sub>b<sub>,</sub> Y<sub>.;</sub> M<sub>anoc</sub>h<sub>a,</sub> D<sub>.; an</sub>d Zh<sub>ou,</sub> T<sub>.</sub> 2024<sub>.</sub> H<sub>a</sub>ll<sub>us</sub>i<sub>on</sub>B<sub>enc</sub>h<sub>:</sub> A<sub>n</sub> Ad<sub>vance</sub>d Di<sub>agnos</sub>ti<sub>c</sub> S<sub>u</sub>it<sub>e</sub> f<sub>or</sub> E<sub>n</sub>t<sub>ang</sub>l<sub>e</sub>d L<sub>anguage</sub> H<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>ti<sub>on an</sub>d Vi<sub>sua</sub>l Ill<sub>u-</sub> sion in Large Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14375–14385.

H<sub>unyuan</sub> Vi<sub>s</sub>i<sub>on</sub> T<sub>eam;</sub> L<sub>yu,</sub> P<sub>.;</sub> W<sub>an,</sub> X<sub>.;</sub> Li<sub>,</sub> G<sub>.;</sub> P<sub>eng,</sub> S<sub>.;</sub> W<sub>ang,</sub> W<sub>.;</sub> W<sub>u,</sub> L<sub>.;</sub> Sh<sub>en,</sub> H<sub>.;</sub> Zh<sub>ou,</sub> Y<sub>.;</sub> T<sub>ang,</sub> C<sub>.; e</sub>t <sub>a</sub>l<sub>.</sub> 2025. HunyuanOCR Technical Report. arXiv preprint arXiv:2511.19575.

Ki<sub>m</sub>i T<sub>eam.</sub> 2026<sub>.</sub> Ki<sub>m</sub>i K3<sub>:</sub> O<sub>pen</sub> F<sub>ron</sub>ti<sub>er</sub> I<sub>n</sub>t<sub>e</sub>lli<sub>gence.</sub> arXiv preprint arXiv:2607.24653.

L<sub>evens</sub>ht<sub>e</sub>i<sub>n,</sub> V<sub>.</sub> I<sub>.</sub> 1966<sub>.</sub> Bi<sub>nary</sub> C<sub>o</sub>d<sub>es</sub> C<sub>apa</sub>bl<sub>e</sub> <sub>o</sub>f C<sub>orrec</sub>ti<sub>ng</sub> Deletions, Insertions, and Reversals. Soviet Physics Doklady, 10(8): 707–710.

Li<sub>,</sub> G<sub>.;</sub> W<sub>an,</sub> X<sub>.;</sub> P<sub>eng,</sub> S<sub>.;</sub> W<sub>ang,</sub> W<sub>.;</sub> F<sub>eng,</sub> H<sub>.;</sub> D<sub>u,</sub> Y<sub>.;</sub> W<sub>u,</sub>B<sub>.;</sub> R<sub>uan,</sub> Z<sub>.;</sub> L<sub>u,</sub> Z<sub>.;</sub> W<sub>u,</sub> L<sub>.;</sub> L<sub>yu,</sub> P<sub>.;</sub> Sh<sub>en,</sub> H<sub>.;</sub> Li<sub>n,</sub> Z<sub>.;</sub> H<sub>u,</sub>

S<sub>.;</sub> Y<sub>ang,</sub> J<sub>.;</sub> W<sub>en,</sub> H<sub>.;</sub> Y<sub>u,</sub> G<sub>.;</sub> Li<sub>u,</sub> H<sub>.;</sub> W<sub>ang,</sub> B<sub>.;</sub> M<sub>a,</sub> C<sub>.;</sub> H<sub>u,</sub> H<sub>.;</sub> Zh<sub>ang,</sub> C<sub>.; an</sub>d Zh<sub>ou,</sub> Y<sub>.</sub> 2026<sub>.</sub> H<sub>unyuan</sub>OCR<sub>-</sub>1<sub>.</sub>5<sub>:</sub> Making Lightweight OCR VLMs Faster and Better. arXiv preprint arXiv:2607.04884.

Li<sub>,</sub> Y<sub>.;</sub> Y<sub>a</sub>n<sub>g,</sub> G<sub>.;</sub> Li<sub>u,</sub> H<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> B<sub>.; a</sub>nd Zh<sub>a</sub>n<sub>g,</sub> C<sub>.</sub> 2025<sub>.</sub> d<sub>o</sub>t<sub>s.ocr:</sub> M<sub>u</sub>ltili<sub>ngua</sub>l D<sub>ocumen</sub>t L<sub>ayou</sub>t P<sub>ars</sub>i<sub>ng</sub> i<sub>n</sub> <sub>a</sub> Si<sub>ng</sub>l<sub>e</sub> Vision-Language Model. arXiv preprint arXiv:2512.02498.

Liu, C.-L.; Yin, F.; Wan<sub>g</sub>, D.-H.; and Wan<sub>g</sub>, Q.-F. 2011. CASIA O<sub>n</sub>li<sub>ne an</sub>d Ofli<sub>ne</sub> Chi<sub>nese</sub> H<sub>an</sub>d<sub>wr</sub>iti<sub>ng</sub> D<sub>a</sub>t<sub>a</sub>b<sub>ases.</sub> In 2011 International Conference on DocumentAnalysis and Recognition, 37–41.

Li<sub>u,</sub> Y<sub>.;</sub> Li<sub>,</sub> Z<sub>.;</sub> H<sub>uang,</sub> M<sub>.;</sub> Y<sub>ang,</sub> B<sub>.;</sub> Y<sub>u,</sub> W<sub>.;</sub> Li<sub>,</sub> C<sub>.;</sub> Yi<sub>n,</sub> X<sub>.;</sub> Li<sub>u,</sub> C<sub>.-</sub>L<sub>.;</sub> Ji<sub>n,</sub> L<sub>.; an</sub>d B<sub>a</sub>i<sub>,</sub> X<sub>.</sub> 2024<sub>.</sub> OCRB<sub>enc</sub>h<sub>:</sub> O<sub>n</sub> th<sub>e</sub> Hidd<sub>en</sub> M<sub>ys</sub>t<sub>ery o</sub>f OCR i<sub>n</sub> L<sub>arge</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l M<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> Science China Information Sciences, 67: 220102.

M<sub>a</sub>hd<sub>av</sub>i<sub>,</sub> M<sub>.;</sub> Z<sub>an</sub>ibbi<sub>,</sub> R<sub>.;</sub> M<sub>ouc</sub>hè<sub>re,</sub> H<sub>.;</sub> Vi<sub>ar</sub>d<sub>-</sub>G<sub>au</sub>di<sub>n,</sub> C<sub>.;</sub> <sub>an</sub>d G<sub>ara</sub>i<sub>n,</sub> U<sub>.</sub> 2019<sub>.</sub> ICDAR 2019 CROHME + TFD<sub>:</sub> C<sub>om-</sub> <sub>pe</sub>titi<sub>on</sub> <sub>on</sub> R<sub>ecogn</sub>iti<sub>on</sub> <sub>o</sub>f H<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> M<sub>a</sub>th<sub>ema</sub>ti<sub>ca</sub>l E<sub>x-</sub> pressions and Typeset Formula Detection. In 2019 International Conference on Document Analysis and Recognition, 1533<sub>–</sub>1538<sub>.</sub>

M<sub>ar</sub>ti<sub>,</sub> U<sub>.-</sub>V<sub>.; an</sub>d B<sub>un</sub>k<sub>e,</sub> H<sub>.</sub> 2002<sub>.</sub> Th<sub>e</sub> IAM<sub>-</sub>D<sub>a</sub>t<sub>a</sub>b<sub>ase:</sub> A<sub>n</sub> E<sub>ng</sub>li<sub>s</sub>h S<sub>en</sub>t<sub>ence</sub> D<sub>a</sub>t<sub>a</sub>b<sub>ase</sub> f<sub>or</sub> Ofli<sub>ne</sub> H<sub>an</sub>d<sub>wr</sub>iti<sub>ng</sub> R<sub>ecog-</sub> nition. International Journal on Document Analysis and Recognition, 5(1): 39–46.

M<sub>a</sub>th<sub>ew,</sub> M<sub>.;</sub> K<sub>ara</sub>t<sub>zas,</sub> D<sub>.; an</sub>d J<sub>awa</sub>h<sub>ar,</sub> C<sub>.</sub> V<sub>.</sub> 2021<sub>.</sub> DocVQA: A Dataset for VQA on Document Ima<sub>g</sub>es. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, 2200–2209.

Ni<sub>u,</sub> J<sub>.;</sub> Li<sub>u,</sub> Z<sub>.;</sub> G<sub>u,</sub> Z<sub>.;</sub> W<sub>ang,</sub> B<sub>.;</sub> O<sub>uyang,</sub> L<sub>.;</sub> Zh<sub>ao,</sub> Z.; Chu, T.; He, T.; Wu, F.; Zhan<sub>g</sub>, Q.; et al. 2025. Mi<sub>ner</sub>U2<sub>.</sub>5<sub>:</sub> A D<sub>ecoup</sub>l<sub>e</sub>d Vi<sub>s</sub>i<sub>on-</sub>L<sub>anguage</sub> M<sub>o</sub>d<sub>e</sub>l f<sub>or</sub> Ef<sub>-</sub> ficient High-Resolution Document Parsing. arXiv preprint arXiv:2509.22186.

O<sub>pen</sub>GVL<sub>a</sub>b<sub>.</sub> 2025<sub>.</sub> I<sub>n</sub>t<sub>ern</sub>VL3<sub>.</sub>5<sub>.</sub> Ofi<sub>c</sub>i<sub>a</sub>l <sub>re</sub>l<sub>ease</sub> bl<sub>og.</sub> A<sub>c-</sub> <sub>cesse</sub>d 2026<sub>-</sub>07<sub>-</sub>27<sub>.</sub>

Ou<sub>y</sub>an<sub>g</sub>, L.; Qu, Y.; Zhou, H.; Zhu, J.; Zhan<sub>g</sub>, R.; Lin, Q.;W<sub>a</sub>n<sub>g,</sub> B<sub>.;</sub> Zh<sub>ao,</sub> Z<sub>.;</sub> Ji<sub>a</sub>n<sub>g,</sub> M<sub>.;</sub> Zh<sub>ao,</sub> X<sub>.;</sub> Shi<sub>,</sub> J<sub>.;</sub> W<sub>u,</sub> F<sub>.;</sub>Ch<sub>u,</sub> P<sub>.;</sub> Li<sub>u,</sub> M<sub>.;</sub> Li<sub>,</sub> Z<sub>.;</sub> X<sub>u,</sub> C<sub>.;</sub> Zh<sub>ang,</sub> B<sub>.;</sub> Shi<sub>,</sub> B<sub>.;</sub> T<sub>u,</sub> Z<sub>.;</sub><sub>an</sub>d H<sub>e,</sub> C<sub>.</sub> 2025<sub>.</sub> O<sub>mn</sub>iD<sub>oc</sub>B<sub>enc</sub>h<sub>:</sub> B<sub>enc</sub>h<sub>mar</sub>ki<sub>ng</sub> Di<sub>verse</sub>PDF D<sub>ocu</sub>m<sub>e</sub>nt P<sub>a</sub>r<sub>s</sub>in<sub>g w</sub>ith C<sub>o</sub>m<sub>p</sub>r<sub>e</sub>h<sub>e</sub>n<sub>s</sub>i<sub>ve</sub> Ann<sub>o</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub>In Proceedings of the IEEE/CVF Conference on ComputerVision and Pattern Recognition, 24838–24848.

Pal, A.; Biswas, S.; Das, A.; Lodh, A.; Banerjee, P.; Chatt<sub>opa</sub>dh<sub>yay,</sub> S<sub>.;</sub> K<sub>a</sub>r<sub>a</sub>t<sub>zas,</sub> D<sub>.;</sub> Ll<sub>a</sub>d<sub>os,</sub> J<sub>.; a</sub>nd J<sub>awa</sub>h<sub>a</sub>r<sub>,</sub> C<sub>.</sub> V<sub>.</sub> 2025<sub>.</sub> N<sub>o</sub>T<sub>e</sub>S<sub>-</sub>B<sub>an</sub>k<sub>:</sub> B<sub>enc</sub>h<sub>mar</sub>ki<sub>ng</sub> N<sub>eura</sub>l T<sub>ranscr</sub>i<sub>p</sub>ti<sub>on an</sub>d Search for Scientific Notes Understanding. arXiv preprint arXiv:2504.09249.

Qwen Team. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

S<sub>eo</sub>n<sub>g,</sub> J<sub>.;</sub> Li<sub>e</sub>rm<sub>a</sub>nn<sub>,</sub> W<sub>.;</sub> Kim<sub>,</sub> M<sub>.;</sub> Shin<sub>,</sub> J<sub>.-</sub>h<sub>.; a</sub>nd Lim<sub>,</sub> S<sub>.</sub> 2026<sub>.</sub> Wh<sub>en</sub> VLM<sub>s</sub> “Fi<sub>x</sub>” St<sub>u</sub>d<sub>en</sub>t<sub>s:</sub> Id<sub>en</sub>tif<sub>y</sub>i<sub>ng</sub> <sub>an</sub>d P<sub>ena</sub>l<sub>-</sub> i<sub>z</sub>i<sub>ng</sub> O<sub>ver-</sub>C<sub>orrec</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on o</sub>f M<sub>u</sub>lti<sub>-</sub>li<sub>ne</sub> H<sub>an</sub>d<sub>-</sub> written Math OCR. arXiv preprint arXiv:2604.22774.

W<sub>ang,</sub> B<sub>.;</sub> H<sub>e,</sub> T<sub>.;</sub> O<sub>uyang,</sub> L<sub>.;</sub> W<sub>u,</sub> F<sub>.;</sub> Zh<sub>ao,</sub> Z<sub>.;</sub> Ch<sub>u,</sub> T<sub>.;</sub> Qu, Y.; Jin, Z.; Zen<sub>g</sub>, W.; Miao, Z.; et al. 2026. MinerU2.5- Pr<sub>o:</sub> P<sub>us</sub>hin<sub>g</sub> th<sub>e</sub> Limit<sub>s</sub> <sub>o</sub>f D<sub>a</sub>t<sub>a-</sub>C<sub>e</sub>ntri<sub>c</sub> D<sub>ocu</sub>m<sub>e</sub>nt P<sub>a</sub>r<sub>s</sub>in<sub>g</sub> at Scale. arXiv preprint arXiv:2604.04771.

W<sub>ang,</sub> B<sub>.;</sub> W<sub>u,</sub> F<sub>.;</sub> O<sub>uyang,</sub> L<sub>.;</sub> G<sub>u,</sub> Z<sub>.;</sub> Zh<sub>ang,</sub> R<sub>.;</sub> Xi<sub>a,</sub> R<sub>.;</sub> Shi<sub>,</sub> B<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> B<sub>.;</sub> <sub>a</sub>nd H<sub>e,</sub> C<sub>.</sub> 2025<sub>.</sub> Im<sub>age</sub> O<sub>ve</sub>r T<sub>ex</sub>t<sub>:</sub> T<sub>rans</sub>f<sub>orm</sub>i<sub>ng</sub> F<sub>ormu</sub>l<sub>a</sub> R<sub>ecogn</sub>iti<sub>on</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on w</sub>ith Ch<sub>arac-</sub> ter Detection Matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19681<sub>–</sub>19690<sub>.</sub>

W<sub>e</sub>i<sub>,</sub> H<sub>.;</sub> S<sub>u</sub>n<sub>,</sub> Y<sub>.; a</sub>nd Li<sub>,</sub> Y<sub>.</sub> 2026<sub>.</sub> D<sub>eep</sub>S<sub>ee</sub>k<sub>-</sub>OCR 2<sub>:</sub> Vi<sub>sua</sub>l Causal Flow. arXiv preprint arXiv:2601.20552.

W<sub>u,</sub> H<sub>.;</sub> L<sub>ou,</sub> H<sub>.;</sub> Li<sub>,</sub> X<sub>.;</sub> Zh<sub>ong,</sub> Z<sub>.;</sub> S<sub>un,</sub> Z<sub>.;</sub> Ch<sub>en,</sub> P<sub>.;</sub> Zh<sub>ou,</sub> X<sub>.;</sub> Z<sub>uo,</sub> K<sub>.;</sub> Ch<sub>en,</sub> Y<sub>.;</sub> T<sub>ang,</sub> X<sub>.;</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2026<sub>.</sub> Fi<sub>re</sub>R<sub>e</sub>d<sub>-</sub>OCR Technical Report. arXiv preprint arXiv:2603.01840.

Yin, Y.; Liu, H.; Xie, Q.; Liu, C.; Yan<sub>g</sub>, S.; Wan<sub>g</sub>, S.; Liu, Z<sub>.;</sub> Z<sub>ou,</sub> H<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> J<sub>.;</sub> W<sub>e</sub>i<sub>,</sub> S<sub>.;</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2026<sub>.</sub> Unlimit<sub>e</sub>d OCR Works. arXiv preprint arXiv:2606.23050.

Zh<sub>ang,</sub> H<sub>.;</sub> Li<sub>ang,</sub> L<sub>.; an</sub>d Ji<sub>n,</sub> L<sub>.</sub> 2020<sub>.</sub> SCUT<sub>-</sub>HCCD<sub>oc:</sub> A N<sub>ew</sub> B<sub>enc</sub>h<sub>mar</sub>k D<sub>a</sub>t<sub>ase</sub>t <sub>o</sub>f H<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> Chi<sub>nese</sub> T<sub>ex</sub>t i<sub>n</sub> U<sub>n-</sub> constrained Camera-Captured Documents. Pattern Recognition, 108: 107559.

Zh<sub>a</sub>n<sub>g,</sub> Z<sub>.;</sub> Li<sub>u,</sub> H<sub>.;</sub> Li<sub>a</sub>n<sub>g,</sub> S<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> Y<sub>.;</sub> Xi<sub>a</sub>n<sub>g,</sub> Y<sub>.;</sub> Li<sub>u,</sub> J<sub>.;</sub> S<sub>u</sub>n<sub>,</sub> T<sub>.;</sub> Lin<sub>,</sub> M<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> Y<sub>.;</sub> Zh<sub>ou,</sub> C<sub>.;</sub> G<sub>ao,</sub> T<sub>.;</sub> C<sub>u</sub>i<sub>,</sub> C<sub>.;</sub> Li<sub>u,</sub> Y<sub>.;</sub> Y<sub>u,</sub> D<sub>.; an</sub>d M<sub>a,</sub> Y<sub>.</sub> 2026<sub>.</sub> P<sub>a</sub>ddl<sub>e</sub>OCR<sub>-</sub>VL<sub>-</sub>1<sub>.</sub>6<sub>:</sub> E<sub>xpan</sub>d<sub>-</sub> i<sub>ng</sub> th<sub>e</sub> F<sub>ron</sub>ti<sub>er o</sub>f D<sub>ocumen</sub>t P<sub>ars</sub>i<sub>ng w</sub>ith U<sub>n</sub>d<sub>er-</sub>O<sub>p</sub>ti<sub>m</sub>i<sub>ze</sub>d Region Refinement and Progressive Post-Training. arXiv preprint arXiv:2606.03264.

Zh<sub>eng,</sub> H<sub>.;</sub> Li<sub>,</sub> Y<sub>.;</sub> Zh<sub>ang,</sub> K<sub>.;</sub> Xi<sub>n,</sub> L<sub>.;</sub> Zh<sub>ao,</sub> G<sub>.;</sub> Li<sub>u,</sub> H<sub>.;</sub> Chen, J.; Lou, J.; Qiu, J.; Fu, Q.; et al. 2026. Multimodal OCR: Parse Anything from Documents. arXiv preprint arXiv:2603.13032.

Zh<sub>e</sub>n<sub>g,</sub> X<sub>.;</sub> B<sub>u</sub>rdi<sub>c</sub>k<sub>,</sub> D<sub>.;</sub> P<sub>opa,</sub> L<sub>.;</sub> Zh<sub>o</sub>n<sub>g,</sub> X<sub>.; a</sub>nd W<sub>a</sub>n<sub>g,</sub> N<sub>.</sub> X. R. 2021. Global Table Extractor (GTE): A Framework f<sub>or</sub> J<sub>o</sub>i<sub>n</sub>t T<sub>a</sub>bl<sub>e</sub> Id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on an</sub>d C<sub>e</sub>ll St<sub>ruc</sub>t<sub>ure</sub> R<sub>ecogn</sub>iti<sub>on</sub> Using Visual Context. In Proceedings of the IEEE/CVF Winter Conference onApplications ofComputer Vision, 697– 706<sub>.</sub>

Zh<sub>ong,</sub> X<sub>.;</sub> Sh<sub>a</sub>fi<sub>e</sub>iB<sub>avan</sub>i<sub>,</sub> E<sub>.; an</sub>d Ji<sub>meno</sub> Y<sub>epes,</sub> A<sub>.</sub> 2020<sub>.</sub> I<sub>mage-</sub>B<sub>ase</sub>d T<sub>a</sub>bl<sub>e</sub> R<sub>ecogn</sub>iti<sub>on:</sub> D<sub>a</sub>t<sub>a,</sub> M<sub>o</sub>d<sub>e</sub>l<sub>, an</sub>d E<sub>va</sub>l<sub>ua-</sub> tion. In Computer Vision – ECCV 2020, 564–580. Springer.

Zh<sub>ou,</sub> C<sub>.;</sub> G<sub>ao,</sub> Z<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> X<sub>.;</sub> G<sub>ao,</sub> T<sub>.;</sub> C<sub>u</sub>i<sub>,</sub> C<sub>.;</sub> T<sub>a</sub>n<sub>g,</sub> J<sub>.;</sub> <sub>a</sub>nd Li<sub>u,</sub> Y<sub>.</sub> 2026<sub>.</sub> R<sub>ea</sub>l5<sub>-</sub>O<sub>mn</sub>iD<sub>oc</sub>B<sub>enc</sub>h<sub>:</sub> A F<sub>u</sub>ll<sub>-</sub>S<sub>ca</sub>l<sub>e</sub> Ph<sub>ys</sub>i<sub>ca</sub>l R<sub>econs</sub>t<sub>ruc</sub>ti<sub>on</sub> B<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> R<sub>o</sub>b<sub>us</sub>t D<sub>ocumen</sub>t P<sub>ars</sub>i<sub>ng</sub> i<sub>n</sub> the Wild. arXiv preprint arXiv:2603.04205.

Zh<sub>u,</sub> Y<sub>.;</sub> Xi<sub>e,</sub> Z<sub>.;</sub> Jin<sub>,</sub> L<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> X<sub>.;</sub> H<sub>ua</sub>n<sub>g,</sub> Y<sub>.; a</sub>nd Zh<sub>a</sub>n<sub>g,</sub> M<sub>.</sub> 2019<sub>.</sub> SCUT<sub>-</sub>EPT<sub>:</sub> N<sub>ew</sub> D<sub>a</sub>t<sub>ase</sub>t <sub>an</sub>d B<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> Of<sub>-</sub> fline Chinese Text Recognition in Examination Paper. IEEE Access, 7: 370–382.

# Supplementary Material

S<sub>ec</sub>ti<sub>on</sub> A <sub>prov</sub>id<sub>es a</sub>dditi<sub>ona</sub>l d<sub>a</sub>t<sub>ase</sub>t <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> b<sub>eyon</sub>d th<sub>ose</sub> i<sub>n</sub> th<sub>e ma</sub>i<sub>n paper.</sub> S<sub>ec</sub>ti<sub>on</sub> B <sub>presen</sub>t<sub>s qua</sub>lit<sub>a</sub>ti<sub>ve</sub> PDE (Prior-Driven Error) exam<sub>p</sub>les from WildHandBench. Section C <sub>g</sub>ives com<sub>p</sub>lete evaluation details includin<sub>g</sub> the full model li<sub>s</sub>t<sub>, promp</sub>t t<sub>emp</sub>l<sub>a</sub>t<sub>es, an</sub>d i<sub>n</sub>f<sub>erence con</sub>fi<sub>gura</sub>ti<sub>on.</sub>

## A. Dataset Details

Thi<sub>s sec</sub>ti<sub>on expan</sub>d<sub>s on</sub> th<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t <sub>cons</sub>t<sub>ruc</sub>ti<sub>on summar</sub>i<sub>ze</sub>d i<sub>n</sub> th<sub>e ma</sub>i<sub>n paper.</sub> W<sub>e repor</sub>t <sub>on</sub>l<sub>y</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s no</sub>t <sub>a</sub>l<sub>rea</sub>d<sub>y covere</sub>d there, including per-source sample counts, the joint category–language distribution, per-scenario counts, and the annotation and i<sub>mage-process</sub>i<sub>ng spec</sub>ifi<sub>ca</sub>ti<sub>ons.</sub>

## A.1 Data Collection Sources

Th<sub>e</sub> 500 <sub>samp</sub>l<sub>es are</sub> d<sub>rawn</sub> f<sub>rom</sub> t<sub>wo comp</sub>l<sub>emen</sub>t<sub>ary c</sub>h<sub>anne</sub>l<sub>s:</sub>

• Ofline handwritten manuscripts (141 samples, 28.2%). Voluntar<sub>y</sub> contributions from <sub>p</sub>roject members and ac<sub>q</sub>uaintances<sub>,</sub> concentrated in the education and medical domains (classroom notes, homework, medical records, <sub>p</sub>rescri<sub>p</sub>tions), <sub>p</sub>lus hand t<sub>ranscr</sub>ib<sub>e</sub>d <sub>ar</sub>ti<sub>c</sub>l<sub>e a</sub>b<sub>s</sub>t<sub>rac</sub>t<sub>s, rea</sub>di<sub>ng no</sub>t<sub>es, an</sub>d l<sub>e</sub>tt<sub>ers</sub> t<sub>o</sub> b<sub>roa</sub>d<sub>en wr</sub>iti<sub>ng s</sub>t<sub>y</sub>l<sub>es.</sub>

• Internet collection (359 samples, 71.8%). Publicl<sub>y</sub> available handwritten ima<sub>g</sub>es coverin<sub>g</sub> lon<sub>g</sub>-tail <sub>g</sub>enres: classical <sub>p</sub>oetr<sub>y</sub> <sub>an</sub>d <sub>ca</sub>lli<sub>grap</sub>h<sub>y,</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>l <sub>arc</sub>hi<sub>va</sub>l <sub>reg</sub>i<sub>s</sub>t<sub>ers, o</sub>ld <sub>newspapers, an</sub>d <sub>o</sub>fi<sub>c</sub>i<sub>a</sub>l f<sub>orms.</sub>

## A.2 Category × Language Distribution

The main paper reports the marginal counts for document structure and language separately. Table 3 gives the full joint di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub>

Table 3: Category × Language joint distribution.
<table><tr><td>Category</td><td>zh_hans</td><td>en</td><td>zh_hant</td><td>zh_en_mix</td><td>Total</td></tr><tr><td>Text</td><td>251</td><td>99</td><td>12</td><td>7</td><td>369</td></tr><tr><td>Table</td><td>62</td><td>16</td><td>0</td><td>1</td><td>79</td></tr><tr><td>Formula</td><td>20</td><td>29</td><td>0</td><td>3</td><td>52</td></tr><tr><td>Total</td><td>333</td><td>144</td><td>12</td><td>11</td><td>500</td></tr></table>

## A.3 Scenario Distribution

Th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> <sub>paper</sub> <sub>enumera</sub>t<sub>es</sub> th<sub>e</sub> <sub>n</sub>i<sub>ne</sub> <sub>scenar</sub>i<sub>os;</sub> T<sub>a</sub>bl<sub>e</sub> 4 <sub>a</sub>dd<sub>s</sub> th<sub>e</sub>i<sub>r</sub> <sub>per-scenar</sub>i<sub>o</sub> <sub>samp</sub>l<sub>e</sub> <sub>coun</sub>t<sub>s.</sub>

T<sub>a</sub>bl<sub>e</sub> 4<sub>:</sub> P<sub>er-scenar</sub>i<sub>o</sub> <sub>samp</sub>l<sub>e</sub> <sub>coun</sub>t<sub>s.</sub>
<table><tr><td>Scenario</td><td>Count</td></tr><tr><td>Literary essays Letters &amp; notes Medical &amp; health Education &amp; learning Business documents</td><td>93 81 62 61 58 55</td></tr><tr><td>Math &amp; science formulas Classical poetry &amp; calligraphy Daily life notes</td><td>41 32</td></tr><tr><td>Historical archives Total</td><td>17 500</td></tr></table>

## A.4 Dificulty Annotation Protocol

Th<sub>e s</sub>i<sub>x</sub> difi<sub>cu</sub>lt<sub>y</sub> di<sub>mens</sub>i<sub>ons</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> th<sub>e ma</sub>i<sub>n paper are eac</sub>h <sub>score</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> b<sub>y</sub> 5 <sub>rev</sub>i<sub>ewers per samp</sub>l<sub>e, an</sub>d <sub>per</sub> di<sub>mens</sub>i<sub>on</sub> <sub>scores</sub> <sub>are</sub> <sub>average</sub>d <sub>across</sub> <sub>rev</sub>i<sub>ewers.</sub> Th<sub>ese</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>suppor</sub>t fi<sub>ne-gra</sub>i<sub>ne</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> b<sub>u</sub>t <sub>are</sub> <sub>no</sub>t <sub>use</sub>d i<sub>n</sub> th<sub>e</sub> <sub>aggrega</sub>t<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>scores.</sub>

## A.5 Annotation Format Details

Be<sub>y</sub>ond the tar<sub>g</sub>et formats stated in the main <sub>p</sub>a<sub>p</sub>er (Markdown for free text, L<sup>A</sup>T<sub>E</sub>X for formulas, HTML <table> for tables), formula regions are additionally annotated with polygon coordinates (closed polygons rather than axis-aligned boxes) to <sub>accommo</sub>d<sub>a</sub>t<sub>e s</sub>l<sub>an</sub>t<sub>e</sub>d h<sub>an</sub>d<sub>wr</sub>itt<sub>en</sub> l<sub>ayou</sub>t<sub>s, w</sub>ith <sub>eac</sub>h <sub>po</sub>l<sub>ygon pa</sub>i<sub>re</sub>d <sub>one-</sub>t<sub>o-one w</sub>ith it<sub>s</sub> LAT<sub>E</sub>X t<sub>ranscr</sub>i<sub>p</sub>ti<sub>on.</sub> T<sub>a</sub>bl<sub>e anno</sub>t<sub>a</sub>ti<sub>ons</sub> carr<sub>y</sub> full structural marku<sub>p</sub> (rowspan, colspan, header hierarch<sub>y</sub>).

## A.6 Image Processing

All i<sub>mages are s</sub>t<sub>ore</sub>d i<sub>n</sub> PNG f<sub>orma</sub>t <sub>w</sub>ith th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng spec</sub>ifi<sub>ca</sub>ti<sub>ons:</sub>

• Maximum file size: 1 MB (<sub>p</sub>alette o<sub>p</sub>timization a<sub>pp</sub>lied where needed).

• EXIF normalization: rotation ta<sub>g</sub>s are a<sub>pp</sub>lied and stri<sub>pp</sub>ed.

• Namin<sub>g</sub> convention: hardhand\_<UUID\_no\_dashes>.png (32-character hex UUID).

## B. Qualitative PDE Examples

##

大记车去，淘浪尽，千正风统人物。故气西边，人是，三国洞赤壁。乱穿空，惊涛拍岸，卷无经雪。

承想上哇当平，山乔的嫁了，雄姿灵发。羽扇乡， 淡知，墙橹灰飞烟灭。同神游，多情笑我，早华 发。人如梦，一每还醇二月。

## Human Baseline

大江东去，淘浪尽，千古风流人物。故垒西边，人道是，三国周赤壁。乱石穿空，惊涛拍岸，？？？雪。江山如画，一时多少豪杰。

## Ground Truth

## 念奴娇·赤壁怀古

## 念奴娇·赤壁怀古

遥想公瑾当年，小乔初嫁了，雄姿英发。羽扇纶 巾，谈笑间，墙橹灰飞烟灭。？？神游，多情应笑 我，早生华发。人生如梦，一尊还酹江月。

## 宋苏轼

遥想公瑾当年，小乔初嫁了，雄姿英发。羽扇纶巾，谈笑间，墙橹灰飞烟灭。故国神游，多情应笑我，早生华发。人生如梦，一尊还酹江月。

## 宋苏轼

大江东去，淘浪尽，千古风流人物。故垒西边，人道是，三国周赤壁。乱石穿空，惊涛拍岸，卷千堆雪。江山如画，一时多少豪杰。

## Gemini-3.1-Pro

## 念奴娇·赤壁怀古

## 宋苏轼

大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。乱石穿空，惊涛拍岸，卷起千堆雪。江山如画，一时多少豪杰。

遥想公瑾当年，小乔初嫁了，雄姿英发。羽扇纶巾，谈笑间，樯橹灰飞烟灭。故国神游，多情应笑我，早生华发。人生如梦，一尊还酹江月。

Figure 3: Classical poem (calligraphy). A handwritten copy of Su Shi’s Chibi Nostalgia contains non-standard character <sub>or</sub>d<sub>er</sub>i<sub>ngs.</sub> G<sub>em</sub>i<sub>n</sub>i “<sub>correc</sub>t<sub>s</sub>” th<sub>e</sub> h<sub>an</sub>d<sub>wr</sub>iti<sub>ng</sub> t<sub>o</sub> th<sub>e memor</sub>i<sub>ze</sub>d t<sub>ex</sub>tb<sub>oo</sub>k <sub>vers</sub>i<sub>on—swapp</sub>i<sub>ng</sub> th<sub>e non-s</sub>t<sub>an</sub>d<sub>ar</sub>d t<sub>wo-c</sub>h<sub>arac</sub>t<sub>er</sub> ordering (táo làng) back to the canonical order (làng táo). This is a classic canonical-text prior: the model overwrites the actual h<sub>an</sub>d<sub>wr</sub>itt<sub>en con</sub>t<sub>en</sub>t <sub>w</sub>ith <sub>a remem</sub>b<sub>ere</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>vers</sub>i<sub>on.</sub> Th<sub>e</sub> h<sub>uman</sub> b<sub>ase</sub>li<sub>ne</sub> f<sub>a</sub>ithf<sub>u</sub>ll<sub>y preserves</sub> th<sub>e non-s</sub>t<sub>an</sub>d<sub>ar</sub>d f<sub>orms an</sub>d <sub>mar</sub>k<sub>s on</sub>l<sub>y</sub> t<sub>ru</sub>l<sub>y</sub> ill<sub>eg</sub>ibl<sub>e c</sub>h<sub>arac</sub>t<sub>ers w</sub>ith “?”<sub>.</sub>

收到这封信你会是什么表情，惊喜还是惊吓？很可惜，暂时只能留给愁一家。

不用检查，这封信没为夹层也没为密文，只是想一看见你意外的表情，所以就写了。人

愚见你时，辛以为又是一七切无聊游戏中的意吓，但后来这境意外却加与了“唯一”的步缀。你的出现打破了很多我过去的信条，隔许久我才发现，这切动落已经影以向到了不少方面，此如多出了你的专展头盔、此如个离开N109区的理还为这一封信。

我尝试过寻找自己不小心深陷的原因，但发现就算找到答案也没为意义，仅仅因为这个人是你。 这个

我不老一把它叫叶做命运，因为你和命這不同，  
命这可以被打不皮，但当我站在你面前，却开  
始期待未来。

## Ground Truth

收到这封信你会是什么表情，惊喜还是惊吓？很可惜，暂时只能留给想象。

不用检查，这封信没有夹层也没有密文，只是想看见你意外的表情，所以就写了。

遇见你时，本以为只是一场无聊游戏中的意外，但后来这场意外却加上了“唯一”的前缀。你的出现打破了很多我过去的信条，隔了许久我才发现，这场动荡已经影响到了不少方面，比如多出了你的专属头盔、比如一个离开N109区的理由，还有这一封信。

我尝试过寻找自己不小心深陷的原因，但发现就算找到答案也没有意义，仅仅因为这个人是你。

我不想把它叫做命运，因为你和命运不同，命运可以被打破，但当我站在你面前，却开始期待未来。秦彻

## Human Baseline

收到这封信你会是什么表情，惊喜还是惊吓？很可惜，暂时只能留给想象。

不用检查，这封信没有夹层也没有密文，只是想看见你意外的表情，所以就写了。

遇见你时，本以为只是一场无聊游戏中的意外，但后来这场意外却加上了“唯一”的前缀。你的出现打破了很多我过去的信条，隔了许久我才发现，这场动荡已经影响到了不少方面，比如多出了你的专属头盔、比如一个离开N109区的理由，还有这一封信。

我尝试过寻找自己不小心深陷的原因，但发现就算找到答案也没有意义，仅仅因为这个人是你。

我不想把它叫做命运，因为你和命运不同，命运可以被打破，但当我站在你面前，却开始期待未来。秦彻

## Gemini-3.1-Pro

收到这封信你会是什么表情，惊喜还是惊吓？很可惜，暂时只能留给想象。

不用检查，这封信没有夹层也没有密文，只是想看见你意外的表情，所以就写了。

遇见你时，本以为又是一场无聊游戏中的意外，但后来这场意外却加上了“唯一”的参数。你的出现打破了很多我过去的信条，隔了许久我才发现，这场动荡已经影响到了不少方面，比如多出了你的专属头盔、比如一个离开N109区的理由，还有这一封信。

我尝试过寻找自己不小心深陷的原因，但发现就算找到答案也没有意义，仅仅因为这个人是你。

我不想把它叫做命运，因为你和命运不同，命运可以被打破，但当我站在你面前，却开始期待未来。秦彻

Fi <sub>ure</sub> 4<sub>:</sub> P<sub>ersona</sub>l l<sub>e</sub>tt<sub>er.</sub> G<sub>em</sub>i<sub>n</sub>i <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>t<sub>es</sub> “ <sub>re</sub>fi<sub>x</sub>” <sub>w</sub>ith “ <sub>arame</sub>t<sub>er</sub>”<sub>—a seman</sub>ti<sub>c-</sub>fi<sub>e</sub>ld <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on</sub> i<sub>n w</sub>hi<sub>c</sub>h th<sub>e mo</sub>d<sub>e</sub> <sub>rep</sub>l<sub>aces</sub> th<sub>e wr</sub>itt<sub>en wor</sub>d <sub>w</sub>ith <sub>a seman</sub>ti<sub>ca</sub>ll<sub>y re</sub>l<sub>a</sub>t<sub>e</sub>d t<sub>erm</sub> d<sub>rawn</sub> f<sub>rom</sub> it<sub>s pr</sub>i<sub>ors ra</sub>th<sub>er</sub> th<sub>an</sub> t<sub>ranscr</sub>ibi<sub>ng</sub> th<sub>e source</sub> f<sub>a</sub>ithf<sub>u</sub>ll<sub>y.</sub>

![](images/1a4bdf224e7f7b0a21eda74da4fc629feeed3bc914ce2629568b8ff5318aff30.jpg)  
Fi<sub>gure</sub> 5<sub>:</sub> Cli<sub>n</sub>i<sub>ca</sub>l di<sub>agnos</sub>i<sub>s.</sub> Th<sub>e</sub> GT di<sub>agnos</sub>i<sub>s</sub> i<sub>s</sub> “<sub>acu</sub>t<sub>e mesen</sub>t<sub>er</sub>i<sub>c</sub> l<sub>ymp</sub>h<sub>a</sub>d<sub>en</sub>iti<sub>s</sub> / <sub>acu</sub>t<sub>e en</sub>t<sub>er</sub>iti<sub>s</sub> / t<sub>erm</sub>i<sub>na</sub>l il<sub>e</sub>iti<sub>s.</sub>” G<sub>em</sub>i<sub>n</sub>i out<sub>p</sub>uts a com<sub>p</sub>letel<sub>y</sub> diferent <sub>y</sub>et medicall<sub>y</sub> <sub>p</sub>lausible dia<sub>g</sub>nosis (“colonic diverticulitis / colonic <sub>p</sub>ol<sub>yp</sub>s / terminal ileitis”; onl<sub>y</sub> the third item matches). This is the most extreme form of PDE: medical domain-knowled<sub>g</sub>e <sub>p</sub>riors fabricate a coherent but f<sub>ac</sub>t<sub>ua</sub>ll<sub>y</sub> <sub>wrong</sub> di<sub>agnos</sub>i<sub>s.</sub> Th<sub>e</sub> h<sub>uman</sub> b<sub>ase</sub>li<sub>ne</sub> h<sub>ones</sub>tl<sub>y</sub> <sub>wr</sub>it<sub>es</sub> “?” f<sub>or</sub> ill<sub>eg</sub>ibl<sub>e</sub> <sub>por</sub>ti<sub>ons.</sub>

![](images/c6929921aaea6feb05ffc5520c26a095f45e4c4b6aa529c3df8b90f13ca74c35.jpg)

## Human Baseline

26年4月26日 中医处方笺门诊号

姓名 穆俊 性别 女 年龄 54 地址穆家店

处方

双花20 连召15 元参15 公英20

花粉10g 皂针10g 牛？12

当归20g 青芍15g ？芥6g

？羌6g 砂花10g 木香10g

茯苓12 炒白术12太子参15

## Ground Truth

26年4月26日 中医处方笺门诊号

姓名 穆俊 性别 女 年龄 54 地址穆家店

处方

双花20 连翘15 元参15 公英20

花粉10g 皂针10g 牛夕12

当归20g 赤芍15g 荆芥6g

川羌6g 砂仁10g 木香10g

茯苓12 炒白术12太子参15

## Gemini-3.1-Pro

26年4月26日 中医处方笺门诊号

姓名 穆俊 性别 女 年龄 54 地址穆家庄

处方

双花20 连召15 元参15 公英20

花粉10g 皂针10g 牛夕12

当归20g 白芍15g 荆芥6g

川羌6g 砂仁10g 桔梗10g

茯苓12 炒白术12 党参15

Fi<sub>gure</sub> 6<sub>:</sub> T<sub>ra</sub>diti<sub>ona</sub>l Chi<sub>nese me</sub>di<sub>c</sub>i<sub>ne prescr</sub>i<sub>p</sub>ti<sub>on.</sub> G<sub>em</sub>i<sub>n</sub>i <sub>app</sub>li<sub>es a pr</sub>i<sub>or-</sub>d<sub>r</sub>i<sub>ven su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on, rep</sub>l<sub>ac</sub>i<sub>ng a</sub> h<sub>er</sub>b <sub>name w</sub>ith <sub>a</sub> <sub>more</sub> f<sub>requen</sub>tl<sub>y prescr</sub>ib<sub>e</sub>d <sub>a</sub>lt<sub>erna</sub>ti<sub>ve.</sub>

## 发样新型举同体制优势开居科技收关

三、增强自主创新能力  
1自更生基点②自主创新必由之路  
③基础创新→源头④国家战略科技力量→着力点  
第四节、加快建设人才强国  
一、培养人才是国家和民族长区发展大计  
①在国家发展中具有重成略地立  
②\~是实现民咳复兴的必然记烃培养就大批练才兼各的高表质人才  
①功以成才，业由才广②走好人生有主培着之把各向优秀人推聚到光和国家事业中来  
一是做好人才21指本传证  
②义是环境社会系中  
留入章发质全过程人民民主  
第一节、坚过中国对色礼合义政治制度自信  
一人民民主是社会合主义的生命  
民主是全人类共同价值  
②民主是共产完和中国人民始终不命生持的重要理念  
二、中国特色礼会主义制度行得适，有生命p.有放率  
根本其本，货重要政经制度组成的制度体系  
②有鲜叫中风色  
三题不移走中国持色社合议政治发展道理路  
①坚持光的导，人风方会作主作法治同有机统一

## Ground Truth

③ 发挥新型举国体制优势开展科技攻关

三、增强自主创新能力

① 自力更生→基点 ② 自主创新→必由之路

③ 基础创新→源头 ④ 国家战略科技力量→着力点

第四节、加快建设人才强国

一、培养人才是国家和民族长远发展大计

① \~在国家发展中具有重要战略地位

②\~是实现民族复兴的必然选择

二、培养造就大批德才兼备的高素质人才

① 功以成才，业由才广

## Human Baseline

## ③ 发挥新型举国体制优势开展科技攻关

## 三、增强自主创新能力

① 自力更生→基点 ② 自主创新→必由之路

③ 基础创新→源头 ④ 国家战略科技力量→着力点

第四节、加快建设人才强国

一、培养人才是国家和民族长远发展大计

①\~在国家发展中具有重要战略地位

②\~是实现民族复兴的必然选择

二、培养造就大批德才兼备的高素质人才

① 功以成才，业由才广

## Gemini-3.1-Pro

## ③ 发挥新型举国体制优势开展科技攻关

## 三、增强自主创新能力

① 自立自强→基点 ② 自主创新→必由之路

③ 基础创新→源头 ④ 国家战略科技力量→着力点

## 第四节、加快建设人才强国

一、培养人才是国家和民族长远发展大计

① ～在国家发展中具有重要战略地位

②\~是实现民族复兴的必然选择

## 二、培养造就大批德才兼备的高素质人才

① 功以才成，业由才广

## PaddleOCR-VL-1.6

## ③ 发挥新型举国体制优势开展科技攻关

## 三、增强自主创新能力

① 自力更生→基点 ② 自主创新→必由之路

③ 基础创新→源头 ④ 国家战略科技力量→着力点

第四节、加快建设人才强国

一、培养人才是国家 and 民族长远发展大计

① \~在国家发展中具有重要战略地位

②\~是实现民族复兴的必然选择

二、培养造就大批德才兼备的高素质人才

① 冲以成才，业由才广

Fi<sub>g</sub>ure 7: Political stud<sub>y</sub> notes. Gemini rewrites into more formal <sub>p</sub>olic<sub>y</sub> terminolo<sub>gy</sub> (“self-reliance” → “self-stren<sub>g</sub>thenin<sub>g</sub>”) and swa s the character order inside a four-character hrase (“achievement makes talent” → “talent makes achievement”)—both lin<sub>g</sub>uisticall<sub>y</sub> valid but not matchin<sub>g</sub> the source. This cleanl<sub>y</sub> contrasts hi<sub>g</sub>h PDE (semantic rewritin<sub>g</sub>) a<sub>g</sub>ainst low PDE (visual misreadin<sub>g</sub>).

![](images/fa6a0f9891f4a0568cf05df175dc8bcd89e7c506cf31c67bffd0e08d08430ada.jpg)  
Fi<sub>gure</sub> 8<sub>:</sub> Child<sub>care</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>re</sub>fl<sub>ec</sub>ti<sub>on.</sub> G<sub>em</sub>i<sub>n</sub>i f<sub>a</sub>b<sub>r</sub>i<sub>ca</sub>t<sub>es</sub> <sub>an</sub> <sub>en</sub>ti<sub>re</sub>l<sub>y</sub> dif<sub>eren</sub>t t<sub>op</sub>i<sub>c:</sub> “d<sub>a</sub>il<sub>y</sub> <sub>c</sub>l<sub>ean</sub>i<sub>ng</sub> <sub>s</sub>kill<sub>s</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub>” b<sub>ecomes</sub> “<sub>recen</sub>t fi<sub>rs</sub>t<sub>-a</sub>id t<sub>ra</sub>i<sub>n</sub>i<sub>ng,</sub>” <sub>an</sub>d “i<sub>n</sub>f<sub>an</sub>t b<sub>a</sub>thi<sub>ng</sub> / <sub>um</sub>bili<sub>ca</sub>l <sub>care</sub>” b<sub>ecomes</sub> “i<sub>n</sub>f<sub>an</sub>t <sub>a</sub>i<sub>rway o</sub>b<sub>s</sub>t<sub>ruc</sub>ti<sub>on care</sub>”<sub>—</sub>th<sub>e w</sub>h<sub>o</sub>l<sub>e passage</sub> to<sub>p</sub>ic is re<sub>p</sub>laced. PaddleOCR-VL exhibits character-level noise but <sub>p</sub>reserves the correct to<sub>p</sub>ic ke<sub>y</sub>words (“dail<sub>y</sub> cleanin<sub>g</sub>,” “bathin<sub>g</sub>”). The human baseline matches the GT exactl<sub>y</sub>.

![](images/499951135100d5fd4b984c4c8e8251f489047152485a196f5db34c005e6f7ea7.jpg)

## Ground Truth

## 忘不了你的善良

在生活中，有许多人，在你看不到的地方，做着温暖人心的事，而在我的心中就有一位那便是我的爷爷。

记得，那是一次寒假的早晨，我揉着睡眼发现爷爷又不见了。我问妈妈，妈妈说爷爷一大早就出门了，我连忙拿起保温杯和围巾就出了家门。

寒风夹着细碎的雪花扑在脸上，远远望见了小区门口的马路上，有一道人影，正挥动着手中的扫把，扫着路上的积雪。我急忙跑过去，爷爷正穿着那件洗得发白的藏青色棉袄。我把保温杯赶紧塞进爷爷那冻红了的手中，他也停下手中动作，拧下杯盖，喝了一小口，白气中露出了一张布满皱纹的溢着幸福的笑脸。

## Human Baseline

## 忘不了你的善良

在生活中，有许多人，在你看不到的地方，做着温暖人心的事，而在我的心中就有一位那便是我的爷爷。

记得，那是一次寒假的早晨，我揉着睡眼发现爷爷又不见了。我问妈妈，妈妈说爷爷一大早就出门了，我连忙拿起保温杯和围巾就出了家门。

寒风夹着细碎的雪花扑在脸上，远远望见了小区门口的马路上，有一道人影，正挥动着手中的扫把，扫着路上的积雪。我急忙跑过去，爷爷正穿着那件洗得发白的藏青色棉袄。我把保温杯赶紧塞进爷爷那冻红了的手中，他也停下手中动作，拧下杯盖，喝了一小口，白气中露出了一张布满皱纹的？？幸福的笑脸。

## Gemini-3.1-Pro

## 忘不了你的善良

在生活中，有许多人，在你最需要帮助的时候，做着温暖人心的事，而在我的心中，就有一位那便是我的爷爷。

记得，那是一次寒假的早晨，我揉着睡眼发现爷爷又不见了。我问妈妈，妈妈说爷爷一大早就出门了，我连忙拿起保温杯和围巾就出家门。

寒风裹挟着细碎的雪花扑在脸上，远远望见了小区门口的马路上，有一道人影，正挥动着手中的扫把，扫着路上的积雪。我急忙跑过去，爷爷正穿着那件洗得发白的藏青色棉袄。我把保温杯赶紧塞进爷爷那冻得通红的手中，他也停下手中动作，拧下杯盖，喝了一小口，白气中露出了布满皱纹的透着幸福的笑脸。

## PaddleOCR-VL-1.6

## 忘不了你的善良

生活中，有许多人，在你生活的旅途中，做着温暖义心的事，而在我的心中，就有一位那便是我的爷爷。

记得，那是一次寒假早早晨，我揉着睑眼发现爷爷又不见了。我问妈妈，妈妈说爷爷一大早就出门了，我连忙拿起保温杯和巾就出家门。

寒风夹着细碎的雪花扑在脸上，远远望见了小区门口的马路上，有一道人影，正挥动着手中的扫把，扫着路上的积雪。我急忙跑过去，爷爷正穿着那件洗得发白的旧色棉织具。我把保温杯托紧进爷爷那冻僵了的手中，他也弯下手中动作，扶下杯盖，喝了一小口，白气中露出了一弓长布满皱纹的幸福的笑脸。

Figure 9: Elementary-school essay. The student wrote “doing heart-warming things in places you cannot see.” Gemini polishes it to “doing heart-warming things when you most need help”—semantically “better” but not the original text (PDE). Both Gemini <sub>an</sub>d P<sub>a</sub>ddl<sub>e</sub>OCR<sub>-</sub>VL <sub>are</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> “<sub>po</sub>li<sub>s</sub>hi<sub>ng</sub>” th<sub>e</sub> <sub>c</sub>hild’<sub>s</sub> <sub>compos</sub>iti<sub>on.</sub>

$$
3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
$$

$$
f ( x , y ) = x ^ { 2 } + 2 y ^ { 2 } \quad , \quad g ( x , y ) = x ^ { 2 } + 1 6 y ^ { 2 } = 1 6
$$

$$
f ( x , y ) = x ^ { 2 } + 2 y ^ { 2 } \quad , \quad g ( x , y ) = x ^ { 2 } + 1 6 y ^ { 2 } = 1 6
$$

we know gradients of both functions will be in the same direction of oportional

$$
\because \angle B A E = \angle A E B = \angle C A E = \angle A
$$

we know gradients of both functions will be in the same direction of

$$
f ( x , y ) = x ^ { 2 } + 2 y ^ { 2 }
$$

$$
f ( x , y ) = x ^ { 2 } + 2 y ^ { 2 }
$$

$$
\because ( - \frac { 1 } { 2 } ) ( - \frac { 1 } { 2 } ) \div ( - \frac { 1 } { 2 } ) = \frac { 1 } { 2 }
$$

$$
\nabla f ( x _ { m } , y _ { m } ) = \lambda \left( \nabla g ( x _ { m } , y _ { m } ) \right)
$$

$$
\nabla f ( x _ { m } , y _ { m } ) = \lambda \left( \nabla g ( x _ { m } , y _ { m } ) \right)
$$

$$
\nabla f = \left( \begin{array} { l } { \partial f } \\ { \partial \tilde { \mathcal { Y } } } \\ { \partial y } \end{array} \right) = \left( \begin{array} { l } { \mathring { \partial } _ { \mathcal { F } } ( x ^ { 2 } + 2 y ^ { 2 } ) } \\ { \mathring { \partial } _ { \mathcal { Y } } ( x ^ { 2 } + 2 y ^ { 2 } ) } \end{array} \right) = \binom { 2 x } { 4 y } = \nabla f
$$

$$
( 1 ) ( - \frac { 1 } { 2 } ) ( - \frac { 1 } { 2 } ) \times ( - \frac { 1 } { 2 } ) \times ( - \frac { 1 } { 2 } )
$$

$$
\nabla f = \left( \begin{array} { l } { \displaystyle \partial f } \\ { \displaystyle \partial f } \\ { \displaystyle \partial y } \end{array} \right) = \left( \begin{array} { l } { \displaystyle \partial _ { \mathcal { F } } ( x ^ { 2 } + 2 y ^ { 2 } ) } \\ { \displaystyle \partial _ { \mathcal { Y } } ( x ^ { 2 } + 2 y ^ { 2 } ) } \end{array} \right) = \left( \begin{array} { l } { \displaystyle 2 x } \\ { \displaystyle 4 y } \end{array} \right) = \nabla f
$$

$$
\nabla g = { \binom { \partial g } { \partial g } } = { \binom { \partial \qquad } { \partial g } } = { \binom { \partial } { \partial g } } ( x ^ { 2 } + 1 6 y ^ { 2 } ) \qquad = { \binom { 2 x } { 3 2 y } }
$$

$$
\nabla g = { \binom { \partial g } { \partial g } } = { \binom { \partial g } { \partial g } } = { \binom { \partial g } { \partial g } } ( x ^ { 2 } + 1 6 y ^ { 2 } ) \Biggr )  = { \binom { 2 x } { 3 2 y } }
$$

$$
\mathrm { E ( 2 m ) } = \mathrm { ( 1 6 0 0 0 0 ) } \times \mathrm { 2 } \times \mathrm { 6 } = \mathrm { 2 } \times \mathrm { 2 } \times \mathrm { 2 }
$$

$$
\mathsf { s o } \parallel \boldsymbol { L } ( \boldsymbol { x } , \boldsymbol { y } , \lambda ) = \nabla f - \lambda \nabla g , \nabla f = \lambda \cdot \nabla g
$$

$$
\mathsf { s o } \parallel \boldsymbol { L } ( \boldsymbol { x } , \boldsymbol { y } , \lambda ) = \nabla f - \lambda \nabla g , \nabla f = \lambda \cdot \nabla g
$$

## Gemini-3.1-Pro

$$
f ( x , y ) = x ^ { 2 } + 2 y ^ { 2 } \quad , \quad g ( x , y ) = x ^ { 2 } + 1 6 y ^ { 2 } = 1 6
$$

We know gradients of both functions will be in the same directior&proportional.

$$
f ( x , y ) = x ^ { 2 } + 2 y ^ { 2 }
$$

## PaddleOCR-VL-1.6

$$
f ( x , y ) = x ^ { 2 } + 2 y ^ { 2 } , g ( x , y ) = x ^ { 2 } + 1 6 y ^ { 2 } = 1 6
$$

We know gradients of both functions will be in the same direction if prepatian

$$
\nabla f ( x _ { m } , y _ { m } ) = \lambda \left( \nabla g ( x _ { m } , y _ { m } ) \right)
$$

$$
f ( x , y ) = x ^ { 2 } + 2 y ^ { 2 }
$$

$$
\nabla f = { \binom { \partial f / \partial x } { \partial f / \partial y } } = { \binom { \partial / \partial x ( x ^ { 2 } + 2 y ^ { 2 } ) } { \partial / \partial y ( x ^ { 2 } + 2 y ^ { 2 } ) } } = { \binom { 2 x } { 4 y } } = \nabla f
$$

$$
\nabla f ( x _ { m } , y _ { m } ) = \lambda ( \nabla g ( x _ { m } , y _ { m } ) )
$$

$$
\nabla f = \left( \begin{array} { l } { \partial f / \partial x } \\ { \partial f / \partial y } \end{array} \right) = \left( \begin{array} { l } { \partial / \partial x ( x ^ { 2 } + 2 y ^ { 2 } ) } \\ { \partial / \partial y ( x ^ { 2 } + 2 y ^ { 2 } ) } \end{array} \right) = \left( \begin{array} { l } { 2 x } \\ { 4 y } \end{array} \right) = \nabla f
$$

$$
\nabla g = { \binom { \partial g / \partial x } { \partial g / \partial y } } = { \binom { \partial / \partial x ( x ^ { 2 } + 1 6 y ^ { 2 } ) } { \partial / \partial y ( x ^ { 2 } + 1 6 y ^ { 2 } ) } } = { \binom { 2 x } { 3 2 y } }
$$

$$
\nabla g \cdot ( \frac { \partial g } { \partial x } ) = \lceil ( \frac { \partial g } { \partial x } x ( x ^ { 2 } + 1 6 y ^ { 2 } ) ) \cdot \lfloor ( \frac { 2 x } { 3 2 y } ) \rceil
$$

$$
\mathrm { s o i f } \ L ( x , y , \lambda ) = f - \lambda g , \quad \nabla f = \lambda \cdot \nabla g
$$

$$
\operatorname { s o } \operatorname { i f } L ( x , y , z ) { \overline { { = \nabla f - \lambda \nabla g , \quad \nabla f } } } = \lambda \cdot \nabla g
$$

Fi<sub>g</sub>ure 10: Mathematical derivation (La<sub>g</sub>ran<sub>g</sub>e multi<sub>p</sub>liers). Gemini corrects the awkward <sub>p</sub>hrase “in the same direction of <sub>propor</sub>ti<sub>ona</sub>l” t<sub>o</sub> “i<sub>n</sub> th<sub>e</sub> <sub>same</sub> di<sub>rec</sub>ti<sub>on</sub> & <sub>propor</sub>ti<sub>ona</sub>l<sub>.</sub>” P<sub>a</sub>ddl<sub>e</sub>OCR<sub>-</sub>VL <sub>m</sub>i<sub>srea</sub>d<sub>s</sub> “<sub>propor</sub>ti<sub>ona</sub>l” <sub>as</sub> “<sub>prepa</sub>ti<sub>an</sub>” <sub>an</sub>d <sub>gar</sub>bl<sub>es</sub> th<sub>e</sub> ∇g matrix line, but shows no semantic rewritin<sub>g</sub>.

T<sub>a</sub>bl<sub>e</sub> 5<sub>:</sub> C<sub>omp</sub>l<sub>e</sub>t<sub>e</sub> li<sub>s</sub>t <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub>
<table><tr><td>Model Name</td><td>Type</td><td>Access</td><td>Model ID / Version</td></tr><tr><td>GPT-5.4</td><td>General VLM</td><td>API (OpenAI-compat)</td><td>gpt-5.4</td></tr><tr><td>Gemini-3.1-Pro</td><td>General VLM</td><td>API (OpenAI-compat)</td><td>gemini-3.1-pro-preview</td></tr><tr><td>Gemini-3.5-Flash</td><td>General VLM</td><td>API (OpenAI-compat)</td><td>gemini-3.5-flash</td></tr><tr><td>Claude-Opus-4-8</td><td>General VLM</td><td>API (OpenAI-compat)</td><td>claude-opus-4-8</td></tr><tr><td>Qwen3.5-Plus</td><td>General VLM</td><td>API (OpenAI-compat)</td><td>qwen3.5-plus</td></tr><tr><td>Qwen3-VL</td><td>General VLM</td><td>API (vLLM)</td><td>qwen3-vl-235b-a22b-thinking</td></tr><tr><td>InternVL3.5</td><td>General VLM</td><td>API</td><td>internv13.5-241b-a28b</td></tr><tr><td>Kimi-K3</td><td>General VLM</td><td>API (Anthropic fmt)</td><td>kimi-k3</td></tr><tr><td>Qianfan-OCR</td><td>OCR-focused</td><td>API (Baidu)</td><td>qianfan-ocr</td></tr><tr><td>Hunyuan-OCR</td><td>OCR-focused</td><td>Local (vLLM)</td><td>tencent/HunyuanOCR</td></tr><tr><td>GLM-OCR</td><td>OCR-focused</td><td>API (ZhipuAI)</td><td>glm-ocr</td></tr><tr><td>PaddleOCR-VL</td><td>OCR-focused</td><td>API (async job)</td><td>PaddleOCR-VL-1.6</td></tr><tr><td>DeepSeek-OCR-v2</td><td>OCR-focused</td><td>Local (vLLM)</td><td>deepseek-ai/DeepSeek-OCR</td></tr><tr><td>FireRed-OCR</td><td>OCR-focused</td><td>Local (vLLM)</td><td>FireRed-OCR</td></tr><tr><td>Unlimited-OCR</td><td>OCR-focused</td><td>Local (vLLM)</td><td>Unlimited-OCR</td></tr><tr><td>MinerU-2.5</td><td>Document parser</td><td>Local (vLLM)</td><td>opendatalab/MinerU2.5-2509-1.2B</td></tr><tr><td>MinerU-2.5-Pro</td><td>Document parser</td><td>Local (vLLM)</td><td>opendatalab/MinerU2.5-Pro</td></tr></table>

## C. Complete Evaluation Details

## C.1 Model List & Access Details

T<sub>a</sub>bl<sub>e</sub> 5 li<sub>s</sub>t<sub>s a</sub>ll <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s w</sub>ith th<sub>e</sub>i<sub>r access me</sub>th<sub>o</sub>d <sub>an</sub>d <sub>mo</sub>d<sub>e</sub>l id<sub>en</sub>tifi<sub>er.</sub>

## C.2 Prompt Templates

All <sub>g</sub>eneral-purpose VLMs (GPT-5.4, Gemini-3.1-Pro, Gemini-3.5-Flash, Claude-Opus-4-8, Qwen3.5-Plus, Qwen3-VL, In ternVL3.5, Kimi-K3) use the standard OmniDocBench <sub>p</sub>rom<sub>p</sub>t. For all OCR-focused models (Qianfan-OCR, Hun<sub>y</sub>uan-OCR, GLM-OCR, PaddleOCR-VL, Dee<sub>p</sub>Seek-OCR-v2, FireRed-OCR, Unlimited-OCR) and document <sub>p</sub>arsers (MinerU-2.5, MinerU 2.5-Pro), we use the default API <sub>p</sub>arameters when callin<sub>g</sub> their APIs, or the default settin<sub>g</sub>s of the model when de<sub>p</sub>lo<sub>y</sub>ed l<sub>oca</sub>ll<sub>y—no user-supp</sub>li<sub>e</sub>d <sub>promp</sub>t i<sub>s overr</sub>idd<sub>en.</sub>

OmniDocBench prompt (default).   
You are an AI assistant specialized in converting PDF   
images to Markdown format. Please follow these   
instructions for the conversion:   
1. Text Processing:   
- Accurately recognize all text content in the PDF   
image without guessing or inferring.   
Convert the recognized text into Markdown format.   
Maintain the original document structure, including   
headings, paragraphs, lists, etc.   
2. Mathematical Formula Processing:   
- Convert all mathematical formulas to LaTeX format.   
Enclose inline formulas with \( \). For example:   
This is an inline formula \( E = mc^2 \)   
Enclose block formulas with \[ \]. For example:   
\[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]   
3. Table Processing:   
- Convert tables to HTML format.   
- Wrap the entire table with <table> and </table>.   
4. Figure Handling:   
- Ignore figures content in the PDF image. Do not   
attempt to describe or convert images.   
5. Output Format:

- Ensure the output Markdown document has a clear   
structure with appropriate line breaks between   
elements.   
- For complex layouts, try to maintain the original   
document’s structure and format as closely as   
possible.   
Please strictly follow these guidelines to ensure   
accuracy and consistency in the conversion. Your task   
is to accurately convert the content of the PDF image   
into Markdown format without adding any extra   
explanations or comments.

## C.3 Inference Configuration

All <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>s</sub>h<sub>are</sub> <sub>a</sub> <sub>common</sub> <sub>re</sub>t<sub>ry</sub> <sub>an</sub>d ti<sub>meou</sub>t <sub>s</sub>t<sub>ra</sub>t<sub>egy:</sub>

• Temperature: 0.0 (deterministic decodin<sub>g</sub>) for all API-based models.

• Max retries: 3<sub>,</sub> with ex<sub>p</sub>onential backof (5 s<sub>,</sub> 10 s<sub>,</sub> 20 s).

• Timeout: 120–300 s de<sub>p</sub>endin<sub>g</sub> on model (u<sub>p</sub> to 600 s for local models).

• Max image dimension: 4000 <sub>p</sub>x (ima<sub>g</sub>es exceedin<sub>g</sub> this are resized <sub>p</sub>ro<sub>p</sub>ortionall<sub>y</sub>).

• Max output tokens: 8192 (where confi<sub>g</sub>urable).

## C.4 Post-Processing

Model out uts are rocessed throu h a clean\_markdown() function that stri s Markdown code fences (e. ., “‘markdown ... “‘) from the res<sub>p</sub>onse. No other normalization is a<sub>pp</sub>lied to model out<sub>p</sub>uts before evaluation.

## C.5 Human Baseline Protocol

Th<sub>e</sub> h<sub>uman</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>s co</sub>ll<sub>ec</sub>t<sub>e</sub>d <sub>un</sub>d<sub>er con</sub>diti<sub>ons</sub> id<sub>en</sub>ti<sub>ca</sub>l t<sub>o mo</sub>d<sub>e</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on:</sub>

• Human <sub>p</sub>artici<sub>p</sub>ants receive the same handwritten document ima<sub>g</sub>es as models.

• Single-pass recognition: no revision<sub>,</sub> no second attem<sub>p</sub>t.

• No auxiliary resources: <sub>p</sub>artici<sub>p</sub>ants ma<sub>y</sub> not consult MLLM <sub>p</sub>redictions<sub>,</sub> writer-<sub>p</sub>rovided transcri<sub>p</sub>ts<sub>,</sub> communit<sub>y</sub> discus-<sub>s</sub>i<sub>ons, or</sub> d<sub>oma</sub>i<sub>n exper</sub>t<sub>s.</sub>

• Same target formats: Markdown for text<sub>,</sub> LAT<sub>E</sub>X for formulas<sub>,</sub> HTML for tables.

• Same evaluation metrics: Edit Distance<sub>,</sub> CDM<sub>,</sub> and TEDS are a<sub>pp</sub>lied identicall<sub>y</sub>.

Thi<sub>s pro</sub>t<sub>oco</sub>l <sub>y</sub>i<sub>e</sub>ld<sub>s a ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>re</sub>f<sub>erence un</sub>d<sub>er mo</sub>d<sub>e</sub>l<sub>-compara</sub>bl<sub>e con</sub>diti<sub>ons ra</sub>th<sub>er</sub> th<sub>an an a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e upper</sub> b<sub>oun</sub>d <sub>on</sub> h<sub>uman</sub> <sub>p</sub>er<sup>f</sup>ormance.