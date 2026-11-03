# Structured Visual Landscape: Generating Preferred Representations in Biological and Artificial Neural Networks

Y<sub>u</sub>son<sub>g</sub> Zhao <sub>yusongz@</sub>illi<sub>no</sub>i<sub>s.e</sub>d<sub>u</sub> Uni<sub>v</sub>ersit<sub>y</sub> of Illinois Urbana-Cham<sub>p</sub>ai<sub>g</sub>n Urb<sub>a</sub>n<sub>a,</sub> Illin<sub>o</sub>i<sub>s,</sub> Unit<sub>e</sub>d St<sub>a</sub>t<sub>es</sub>

Hao Wan<sub>g</sub> h<sub>wangm</sub>l<sub>@</sub>illi<sub>no</sub>i<sub>s.e</sub>d<sub>u</sub> Uni<sub>v</sub>ersit<sub>y</sub> of Illinois Urbana-Cham<sub>p</sub>ai<sub>g</sub>n Urb<sub>a</sub>n<sub>a,</sub> Illin<sub>o</sub>i<sub>s,</sub> Unit<sub>e</sub>d St<sub>a</sub>t<sub>es</sub>

## Abstract

Understandin<sub>g</sub> how neurons res<sub>p</sub>ond to visual stimulus is an im-<sub>p</sub>ortant <sub>q</sub>uestion in both dee<sub>p</sub> learnin<sub>g</sub> and neuroscience. It has si<sub>g</sub>nificant im<sub>p</sub>lications for enhancin<sub>g</sub> the inter<sub>p</sub>retabilit<sub>y</sub> of black b<sub>ox ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>s an</sub>d f<sub>or un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> th<sub>e v</sub>i<sub>sua</sub>l re<sub>p</sub>resentation in the human brain. We <sub>p</sub>ro<sub>p</sub>ose a structured visual re<sub>p</sub>resentations landsca<sub>p</sub>e and desi<sub>g</sub>n an activation-based <sub>p</sub>rior that <sub>a</sub>ll<sub>ows us</sub> t<sub>o e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y regu</sub>l<sub>ar</sub>i<sub>ze</sub> th<sub>e</sub> l<sub>an</sub>d<sub>scape w</sub>ith <sub>ac</sub>ti<sub>va</sub>ti<sub>ons</sub> <sup>f</sup>rom brain regions or neura<sup>l</sup> networ<sup>k</sup> units. Our mode<sup>l</sup> Vis-Lens i<sub>n</sub>t<sub>egra</sub>t<sub>es a var</sub>i<sub>a</sub>ti<sub>ona</sub>l <sub>au</sub>t<sub>oenco</sub>d<sub>er an</sub>d <sub>a</sub> dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l i<sub>n</sub>t<sub>o an</sub> i<sub>mage genera</sub>ti<sub>on</sub> f<sub>ramewor</sub>k<sub>.</sub> It <sub>ena</sub>bl<sub>es</sub> th<sub>e genera</sub>ti<sub>on o</sub>f <sub>na</sub>t<sub>ura</sub>l<sub>,</sub> realistic <sub>p</sub>referred ima<sub>g</sub>es b<sub>y</sub> directl<sub>y</sub> modif<sub>y</sub>in<sub>g</sub> the activationre<sub>g</sub>ularized latent codes without <sub>p</sub>er-ima<sub>g</sub>e iterative o<sub>p</sub>timization. W<sub>e</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>e</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>veness</sub> <sub>o</sub>f <sub>our</sub> f<sub>ramewor</sub>k <sub>on</sub> b<sub>o</sub>th <sub>ar</sub>ti<sub>-</sub> fi<sub>c</sub>i<sub>a</sub>l <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>s</sub> <sub>an</sub>d h<sub>uman</sub> <sub>v</sub>i<sub>sua</sub>l <sub>cor</sub>t<sub>ex</sub> <sub>us</sub>i<sub>ng</sub> f<sub>unc</sub>ti<sub>ona</sub>l ma<sub>g</sub>netic resonance ima<sub>g</sub>in<sub>g</sub> (fMRI) data. Our framework im<sub>p</sub>roves activation modulation and ima<sub>g</sub>e realism com<sub>p</sub>ared with a recent <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t <sub>me</sub>th<sub>o</sub>d f<sub>or</sub> <sub>genera</sub>ti<sub>ng</sub> <sub>pre</sub>f<sub>erre</sub>d <sub>v</sub>i<sub>sua</sub>l <sub>represen</sub>t<sub>a-</sub> ti<sub>ons o</sub>f th<sub>ese neura</sub>l t<sub>arge</sub>t<sub>s.</sub>

## Keywords

visual re<sub>p</sub>resentation<sub>,</sub> brain decodin<sub>g,</sub> variational autoencoder<sub>,</sub> <sub>p</sub>ref<sub>erre</sub>d i<sub>mages, neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>s,</sub> f<sub>ea</sub>t<sub>ure v</sub>i<sub>sua</sub>li<sub>za</sub>ti<sub>on</sub>

## 1 Introduction

Th<sub>e exp</sub>l<sub>ora</sub>ti<sub>on o</sub>f <sub>v</sub>i<sub>sua</sub>l <sub>represen</sub>t<sub>a</sub>ti<sub>ons</sub> i<sub>n</sub> b<sub>o</sub>th bi<sub>o</sub>l<sub>og</sub>i<sub>ca</sub>l <sub>an</sub>d <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>s</sub> h<sub>as a</sub>d<sub>vance</sub>d <sub>our un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng o</sub>f h<sub>ow</sub> <sub>comp</sub>l<sub>ex v</sub>i<sub>sua</sub>l <sub>s</sub>ti<sub>mu</sub>li <sub>are processe</sub>d <sub>an</sub>d <sub>represen</sub>t<sub>e</sub>d i<sub>n neura</sub>l networks [4, 27, 41]. Previous studies have demonstrated the ca-<sub>p</sub>acit<sub>y</sub> of neurons in the medial tem<sub>p</sub>oral lobe (MTL) of the human brain to form invariant re<sub>p</sub>resentations of com<sub>p</sub>lex stimuli [63], suc<sup>h</sup> as <sup>f</sup>aces, <sup>l</sup>an<sup>d</sup>mar<sup>k</sup>s, an<sup>d</sup> o<sup>b</sup>jects, regar<sup>dl</sup>ess o<sup>f</sup> visua<sup>l</sup> vari-<sub>a</sub>ti<sub>ons.</sub> Th<sub>ese</sub> fi<sub>n</sub>di<sub>ngs</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub> h<sub>ow s</sub>i<sub>ng</sub>l<sub>e neurons can enco</sub>d<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>l<sub>eve</sub>l<sub>,</sub> <sub>a</sub>b<sub>s</sub>t<sub>rac</sub>t <sub>percep</sub>t<sub>s</sub> <sub>w</sub>ith <sub>remar</sub>k<sub>a</sub>bl<sub>e</sub> <sub>spec</sub>ifi<sub>c</sub>it<sub>y,</sub> l<sub>ea</sub>di<sub>ng</sub> t<sub>o</sub> the h<sub>yp</sub>othesis of s<sub>p</sub>arse and invariant codin<sub>g</sub> mechanisms in the brain [56].

While methods ex<sub>p</sub>lorin<sub>g</sub> visual re<sub>p</sub>resentation in some brain <sub>reg</sub>i<sub>ons</sub> <sub>wor</sub>k fi<sub>ne</sub> b<sub>y</sub> l<sub>earn</sub>i<sub>ng</sub> di<sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub> <sub>o</sub>f <sub>popu-</sub> lar ima<sub>g</sub>e encoders like CLIP [17, 37], understandin<sub>g</sub> the feature <sub>represen</sub>t<sub>a</sub>ti<sub>ons</sub> f<sub>or many</sub> b<sub>ra</sub>i<sub>n reg</sub>i<sub>ons</sub> th<sub>a</sub>t <sub>are no</sub>t <sub>we</sub>ll<sub>-s</sub>t<sub>u</sub>di<sub>e</sub>d remains challen<sub>g</sub>in<sub>g</sub>. For exam<sub>p</sub>le<sub>,</sub> some hi<sub>g</sub>her-order brain re<sub>g</sub>ions

Ed<sub>g</sub>ar Y. Walker <sub>eywa</sub>lk<sub>er@uw.e</sub>d<sub>u</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f W<sub>as</sub>hin<sub>g</sub>t<sub>o</sub>n S<sub>ea</sub>ttl<sub>e,</sub> W<sub>as</sub>hin<sub>g</sub>t<sub>o</sub>n<sub>,</sub> Unit<sub>e</sub>d St<sub>a</sub>t<sub>es</sub>

Mi L<sub>u</sub> l<sub>m</sub>i7<sub>@ga</sub>t<sub>ec</sub>h<sub>.e</sub>d<sub>u</sub>   
Geor<sub>g</sub>ia Institute of Technolo<sub>gy</sub>   
Atl<sub>a</sub>nt<sub>a,</sub> G<sub>eo</sub>r<sub>g</sub>i<sub>a,</sub> Unit<sub>e</sub>d St<sub>a</sub>t<sub>es</sub>

can demonstrate mixed selectivit<sub>y</sub> [65], where the<sub>y</sub> can res<sub>p</sub>ond t<sub>o</sub> i<sub>mages</sub> <sub>w</sub>ith <sub>sma</sub>ll <sub>over</sub>l<sub>ap</sub> i<sub>n</sub> <sub>s</sub>h<sub>are</sub>d f<sub>ea</sub>t<sub>ures.</sub> A<sub>s</sub> <sub>presen</sub>t<sub>e</sub>d i<sub>n</sub> Fi<sub>g</sub>ure 1<sub>,</sub> the to<sub>p</sub> ima<sub>g</sub>es that maximall<sub>y</sub> activate the brain re<sub>g</sub>ion are distributed across the embeddin<sub>g</sub> s<sub>p</sub>ace of the CLIP encoder [64].

Additionall<sub>y,</sub> brain activit<sub>y</sub> measured is often nois<sub>y,</sub> low-resolution<sub>,</sub> <sub>an</sub>d <sub>par</sub>ti<sub>a</sub>ll<sub>y o</sub>b<sub>serve</sub>d<sub>, w</sub>hi<sub>c</sub>h i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> difi<sub>cu</sub>lti<sub>es</sub> f<sub>or ana</sub>l<sub>ys</sub>i<sub>s</sub> and modelin<sub>g</sub>. Unlike <sub>p</sub>revious works that aim to reconstruct entire ima<sub>g</sub>es from brain activities [25, 48, 53, 71], our work focuses on understandin<sub>g</sub> the feature re<sub>p</sub>resentation of brain re<sub>g</sub>ions b<sub>y g</sub>ener-<sub>a</sub>tin<sub>g</sub> n<sub>ew p</sub>r<sub>e</sub>f<sub>e</sub>rr<sub>e</sub>d im<sub>ages</sub> th<sub>a</sub>t m<sub>a</sub>ximiz<sub>e o</sub>r minimiz<sub>e</sub> th<sub>e ac</sub>ti<sub>v</sub>it<sub>y</sub> of those re<sub>g</sub>ions. These s<sub>y</sub>nthetic visual stimuli can be de<sub>p</sub>lo<sub>y</sub>ed in f<sub>o</sub>ll<sub>ow-up neurosc</sub>i<sub>ence exper</sub>i<sub>men</sub>t<sub>s</sub> t<sub>o</sub> t<sub>es</sub>t <sub>new</sub> h<sub>ypo</sub>th<sub>eses, re</sub>fi<sub>ne</sub> ROI functional ma<sub>p</sub>s<sub>,</sub> and accelerate discover<sub>y</sub> b<sub>y</sub> <sub>g</sub>uidin<sub>g</sub> stimulus desi<sub>g</sub>n.

To achieve this <sub>g</sub>oal<sub>,</sub> we <sub>p</sub>ro<sub>p</sub>ose Vis-Lens<sub>,</sub> an efective a<sub>pp</sub>roach to inter<sub>p</sub>ret the feature re<sub>p</sub>resented b<sub>y</sub> brain re<sub>g</sub>ions. Given the nois<sub>y</sub> nat<sub>u</sub>re of brain recordin<sub>g</sub>s<sub>,</sub> instead of <sub>u</sub>sin<sub>g</sub> brain acti<sub>v</sub>ities di<sub>rec</sub>tl<sub>y as</sub> i<sub>npu</sub>t t<sub>o</sub> th<sub>e</sub> d<sub>eco</sub>di<sub>ng mo</sub>d<sub>e</sub>l<sub>, our me</sub>th<sub>o</sub>d <sub>re</sub>fi<sub>nes</sub> th<sub>e</sub> landsca<sub>p</sub>e of visual re<sub>p</sub>resentation b<sub>y</sub> em<sub>p</sub>lo<sub>y</sub>in<sub>g</sub> brain activation to re<sub>g</sub>ularize the <sub>p</sub>rior<sub>,</sub> servin<sub>g</sub> as a soft constraint. We further int<sub>egra</sub>t<sub>e</sub> th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>an</sub>d<sub>scape</sub> <sub>w</sub>ith <sub>a</sub> <sub>var</sub>i<sub>a</sub>ti<sub>ona</sub>l <sub>au</sub>t<sub>o-enco</sub>d<sub>er</sub> (VAE) and difusion <sub>g</sub>enerative models to <sub>g</sub>enerate realistic <sub>p</sub>ref<sub>erre</sub>d i<sub>mages</sub> th<sub>a</sub>t <sub>can e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y mo</sub>d<sub>u</sub>l<sub>a</sub>t<sub>e</sub> t<sub>arge</sub>t b<sub>ra</sub>i<sub>n ac</sub>ti<sub>va</sub>ti<sub>ons.</sub>

O<sub>ur</sub> <sub>me</sub>th<sub>o</sub>d <sub>can</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>exp</sub>l<sub>ora</sub>ti<sub>on</sub> <sub>o</sub>f <sub>v</sub>i<sub>sua</sub>l <sub>represen</sub>t<sub>a-</sub> ti<sub>o</sub>n<sub>s</sub> <sub>o</sub>f br<sub>a</sub>in r<sub>eg</sub>i<sub>o</sub>n<sub>s</sub> <sub>w</sub>hil<sub>e</sub> m<sub>a</sub>int<sub>a</sub>inin<sub>g</sub> int<sub>e</sub>r<sub>p</sub>r<sub>e</sub>t<sub>a</sub>bilit<sub>y.</sub> It i<sub>s</sub> <sub>va</sub>lidated with fMRI data from the Natural Scenes Dataset (NSD) [2], anal<sub>y</sub>zin<sub>g</sub> res<sub>p</sub>onses in the human visual cortex to com<sub>p</sub>lex natural stimuli. Meanwhile<sub>,</sub> due to the scarcit<sub>y</sub> of ima<sub>g</sub>e res<sub>p</sub>onse data from h<sub>uman</sub> b<sub>ra</sub>i<sub>ns, our propose</sub>d f<sub>ramewor</sub>k i<sub>s a</sub>l<sub>so ex</sub>t<sub>en</sub>d<sub>e</sub>d t<sub>o ar</sub>tifi<sub>-</sub> cial neural networks (ANNs), allowin<sub>g</sub> controlled testin<sub>g</sub> of visual feature re<sub>p</sub>resentations and inter<sub>p</sub>retabilit<sub>y</sub> in artificial s<sub>y</sub>stems. B<sub>y</sub> i<sub>n</sub>t<sub>egra</sub>ti<sub>ng</sub> i<sub>ns</sub>i<sub>g</sub>ht<sub>s</sub> f<sub>rom</sub> b<sub>o</sub>th bi<sub>o</sub>l<sub>og</sub>i<sub>ca</sub>l <sub>an</sub>d <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>neura</sub>l <sub>ne</sub>t<sub>-</sub> <sub>wor</sub>k<sub>s, we a</sub>i<sub>m</sub> t<sub>o</sub> b<sub>r</sub>id<sub>ge</sub> th<sub>e un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng o</sub>f <sub>v</sub>i<sub>sua</sub>l <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> strate<sub>g</sub>ies across natural and en<sub>g</sub>ineered s<sub>y</sub>stems. Our contributions <sub>can</sub> b<sub>e ou</sub>tli<sub>ne</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

• We introduce an activation-re<sub>g</sub>ularized <sub>p</sub>rior for VAE [30] that or<sub>g</sub>anizes its latent s<sub>p</sub>ace into two distinct clusters<sub>,</sub> corres<sub>p</sub>ondin<sub>g</sub> to codes that increase or decrease a tar<sub>g</sub>et brain re<sub>g</sub>ion’s activation. This desi<sub>g</sub>n enables eas<sub>y</sub> <sub>p</sub>lu<sub>g</sub>- and-<sub>p</sub>la<sub>y</sub> inte<sub>g</sub>ration with conventional <sub>g</sub>enerative models.

• Our method s<sub>y</sub>nthesizes visual re<sub>p</sub>resentations from s<sub>p</sub>ecifi<sub>e</sub>d b<sub>ra</sub>i<sub>n</sub> <sub>reg</sub>i<sub>ons,</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> b<sub>e</sub>tt<sub>er</sub> <sub>per</sub>f<sub>ormance</sub> t<sub>o</sub> <sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub> b<sub>ra</sub>i<sub>n ac</sub>ti<sub>va</sub>ti<sub>ons</sub> th<sub>an a recen</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t b<sub>ase</sub>li<sub>ne, es-</sub> <sub>p</sub>eciall<sub>y</sub> for decreasin<sub>g</sub> activations. It could also <sub>g</sub>enerate <sub>p</sub>referred visual in<sub>p</sub>uts with better <sub>p</sub>erce<sub>p</sub>tual realism. More-<sub>over,</sub> it <sub>prov</sub>id<sub>es a genera</sub>li<sub>ze</sub>d f<sub>ramewor</sub>k <sub>w</sub>hi<sub>c</sub>h <sub>s</sub>h<sub>ows</sub> ro<sup>b</sup>ustness on cross-su<sup>b</sup>ject <sup>b</sup>io<sup>l</sup>ogica<sup>l d</sup>ata, an<sup>d</sup> can a<sup>l</sup>so b<sub>e</sub> t<sub>rans</sub>f<sub>erre</sub>d t<sub>o</sub> <sub>v</sub>i<sub>sua</sub>li<sub>z</sub>i<sub>ng</sub> f<sub>ea</sub>t<sub>ures</sub> i<sub>n</sub> <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>s.</sub>

![](images/b1691245f99da0907281e35be20a87e02072304569bb0bc04b67195f34f5a24b.jpg)  
Figure 1: Original CLIP Space (Top): Original CLIP embedding space without any activation constraint: image embeddings that elicit high and low neural responses are dispersed and not clustered throughout the space without shared visual structure. Structured Landscape (Bottom): By encoding CLIP embeddings into the regularized latent space constrained with neuronal activations (after z-score normalization) as the prior, our method Vis-Lens could provide a more structured and representative landscape to better identify the representations and preferences of brain regions than sorting top-k images in a dataset. The l b d b f h d [ ] h l d l h l b d d h activation of the target brain region given diferent images as inputs, while the x and y axes represent two latent dimensions, respectively.

• B<sub>y g</sub>eneratin<sub>g</sub> controllable, hi<sub>g</sub>h-<sub>q</sub>ualit<sub>y</sub> visual stimuli that <sub>can</sub> <sub>mo</sub>d<sub>u</sub>l<sub>a</sub>t<sub>e</sub> <sub>spec</sub>ifi<sub>c</sub> <sub>reg</sub>i<sub>ons,</sub> <sub>our</sub> f<sub>ramewor</sub>k d<sub>emons</sub>t<sub>ra</sub>t<sub>es</sub> new <sub>p</sub>otential in vision-based brain com<sub>p</sub>uter interface a<sub>p</sub>- <sub>p</sub>li<sub>ca</sub>ti<sub>ons</sub> <sub>an</sub>d <sub>can</sub> b<sub>e</sub> <sub>use</sub>d t<sub>o</sub> <sub>genera</sub>t<sub>e</sub> <sub>new</sub> h<sub>ypo</sub>th<sub>eses</sub> f<sub>or</sub> f<sub>u</sub>t<sub>ure</sub> <sub>neurosc</sub>i<sub>ence</sub> <sub>s</sub>t<sub>u</sub>di<sub>es.</sub>

## 2 Related Works

Image generative models. Advances in generative mode<sup>l</sup>s <sup>h</sup>ave si<sub>g</sub>nificantl<sub>y</sub> transformed ima<sub>g</sub>e s<sub>y</sub>nthesis. Foundational a<sub>pp</sub>roaches such as GANs [18] led to extensions like Wasserstein GANs [3] and Conditional GANs [43], with fidelit<sub>y</sub> im<sub>p</sub>rovements usin<sub>g</sub> s<sub>p</sub>ectral normalization [44], while VAEs [30] introduced <sub>p</sub>robabilistic encodin<sub>g</sub>, later combined with latent s<sub>p</sub>ace re<sub>g</sub>ularization [24, 77]. Further, Cascaded <sub>g</sub>eneration [24] and autore<sub>g</sub>ressive modelin<sub>g</sub> [9] enhanced s<sub>y</sub>nthesis <sub>q</sub>ualit<sub>y</sub>. More su<sub>p</sub>erior <sub>g</sub>eneration models like Difusion models [23, 67] advanced controllable, hi<sub>g</sub>h-fidelit<sub>y</sub> s<sub>y</sub>nthesis, inte<sub>g</sub>ratin<sub>g</sub> structured <sub>p</sub>erturbations via SDEdit [42] and text-to-ima<sub>g</sub>e ali<sub>g</sub>nment throu<sub>g</sub>h CLIP [64]. IP-Ada<sub>p</sub>ter [83] refined latent s<sub>p</sub>aces for semantic consistenc<sub>y</sub>, while GLIDE [52] and Stable Difusion [67] further demonstrated <sub>p</sub>hotorealistic, text-<sub>g</sub>uided i<sub>mage</sub> <sub>genera</sub>ti<sub>on.</sub> O<sub>ur</sub> <sub>wor</sub>k i<sub>s</sub> d<sub>eve</sub>l<sub>ope</sub>d b<sub>ase</sub>d <sub>on</sub> th<sub>ese</sub> <sub>conven-</sub> ti<sub>o</sub>n<sub>a</sub>l <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>ve</sub> m<sub>o</sub>d<sub>e</sub>l fr<sub>a</sub>m<sub>ewo</sub>rk<sub>s,</sub> it <sub>a</sub>d<sub>op</sub>t<sub>s</sub> VAE<sub>s</sub> t<sub>o</sub> b<sub>u</sub>ild <sub>up</sub> <sub>our</sub> <sub>propose</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>an</sub>d<sub>scape</sub> <sub>an</sub>d <sub>a</sub> St<sub>a</sub>bl<sub>e</sub> Dif<sub>us</sub>i<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l <sub>g</sub>eneration <sub>p</sub>i<sub>p</sub>eline to s<sub>y</sub>nthesize <sub>p</sub>referred ima<sub>g</sub>es conditioned on CLIP <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub> d<sub>er</sub>i<sub>ve</sub>d f<sub>rom</sub> th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>an</sub>d<sub>scape.</sub>

Feature visualization. Visua<sup>l</sup>ization tec<sup>h</sup>niques provide deep insi<sub>g</sub>hts into neural network inter<sub>p</sub>retabilit<sub>y</sub> and functionalit<sub>y,</sub> s<sub>p</sub>anning both biological and artificial systems. Early work by Erhan et al. [15] focused on minimal regularization approaches, while adversarial exam<sub>p</sub>les ex<sub>p</sub>osed vulnerabilities in neural networks [74]. Mahendran and Vedaldi [39] and Nguyen et al. [50] further ex-<sub>p</sub>anded on inter<sub>p</sub>retabilit<sub>y</sub> with total variation re<sub>g</sub>ularization<sub>,</sub> counterexam<sub>p</sub>les<sub>,</sub> and ima<sub>g</sub>e blurrin<sub>g</sub>. Meanwhile<sub>,</sub> techni<sub>q</sub>ues such as DeepDream by Mordvintsev et al. [45, 46], employing jitter, multiscale visualization, and <sub>g</sub>radient normalization [54, 76], enhanced clarity in feature visualization. For prior related work, Nguyen et al. [51] demonstrated generative adversarial synthesis of preferred in<sub>p</sub>uts<sub>,</sub> em<sub>p</sub>hasizin<sub>g</sub> controlled ima<sub>g</sub>e <sub>g</sub>eneration throu<sub>g</sub>h learned <sub>p</sub>riors. Our work extends this line b<sub>y g</sub>eneratin<sub>g</sub> ima<sub>g</sub>es with <sub>p</sub>ref<sub>erre</sub>d f<sub>ea</sub>t<sub>ures</sub> b<sub>ase</sub>d <sub>on pre-</sub>d<sub>e</sub>fi<sub>ne</sub>d <sub>pr</sub>i<sub>ors</sub> i<sub>n</sub> l<sub>an</sub>d<sub>scape s</sub>h<sub>ape</sub>d b<sub>y</sub> activations.

Brain decoding and most exciting images. Generative models levera<sub>g</sub>in<sub>g</sub> brain activit<sub>y</sub> as direct in<sub>p</sub>ut have substantiall<sub>y</sub> advanced neural-to-visual s<sub>y</sub>nthesis. Direct neural-to-ima<sub>g</sub>e s<sub>y</sub>nthesis methods convert fMRI activit<sub>y</sub> into hi<sub>g</sub>h-fidelit<sub>y p</sub>ictures with dif<sub>us</sub>i<sub>o</sub>n <sub>o</sub>r GAN b<sub>ac</sub>kb<sub>o</sub>n<sub>es,</sub> lik<sub>e</sub> Br<sub>a</sub>inDif<sub>us</sub>i<sub>o</sub>n <sub>a</sub>nd N<sub>eu</sub>r<sub>o</sub>DM [37, 61], Brain2GAN [11], latent-difusion variants [16, 58], and ener<sub>gy</sub>-<sub>g</sub>uided a<sub>pp</sub>roaches [60], while activation-o<sub>p</sub>timized methods like Ince<sub>p</sub>tion Loo<sub>p</sub>s [80] and NeuroGen [19] reveal neuronal tunin<sub>g</sub> b<sub>y</sub> iteratin<sub>g</sub> stimuli based on neural res<sub>p</sub>onses. Recent brain decodin<sub>g</sub> studies have continued to im<sub>p</sub>rove visual reconstruction <sub>q</sub>ualit<sub>y</sub> throu<sub>g</sub>h stron<sub>g</sub>er <sub>p</sub>riors<sub>,</sub> better ali<sub>g</sub>nment s<sub>p</sub>aces<sub>,</sub> and cross-su<sup>b</sup>ject or <sup>d</sup>ata-e<sup>fi</sup>cient <sup>f</sup>ormu<sup>l</sup>ations, inc<sup>l</sup>u<sup>d</sup>ing generative decodin<sub>g</sub> with latent difusion [58, 75], difusion-<sub>p</sub>rior-based reconstruction [69, 70], <sub>p</sub>robabilistic visual-to-fMRI s<sub>y</sub>nthesis [40], and cross-subject decodin<sub>g</sub> frameworks [81]. Recent multi-modal a<sub>pp</sub>roaches [6, 14, 26, 28, 62, 79] inte<sub>g</sub>rate <sub>g</sub>enerative models for im r<sub>ove</sub>d <sub>s</sub> nth<sub>es</sub>i<sub>s</sub> r<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n<sub>, w</sub>hil<sub>e</sub> C<sub>o</sub>ntr<sub>o</sub>ll<sub>a</sub>bl<sub>e</sub> Mind Vi<sub>sua</sub>l Diffusion [84], Seein<sub>g</sub> Be<sub>y</sub>ond the Brain [10], and Reconstructin<sub>g</sub> the Mind’s E<sub>y</sub>e [69] levera<sub>g</sub>e structured difusion <sub>p</sub>riors for enh<sub>ance</sub>d <sub>v</sub>i<sub>sua</sub>l <sub>recons</sub>t<sub>ruc</sub>ti<sub>ons.</sub> B<sub>eyon</sub>d <sub>recons</sub>t<sub>ruc</sub>ti<sub>on,</sub> <sub>recen</sub>t <sub>me</sub>th<sub>-</sub> <sub>o</sub>d<sub>s</sub> h<sub>ave a</sub>l<sub>so exp</sub>l<sub>ore</sub>d <sub>r</sub>i<sub>c</sub>h<sub>er a</sub>li<sub>gnmen</sub>t <sub>spaces</sub> b<sub>e</sub>t<sub>ween</sub> b<sub>ra</sub>i<sub>n s</sub>i<sub>g-</sub> <sub>na</sub>l<sub>s</sub> <sub>an</sub>d f<sub>oun</sub>d<sub>a</sub>ti<sub>on-mo</sub>d<sub>e</sub>l <sub>represen</sub>t<sub>a</sub>ti<sub>ons,</sub> <sub>suc</sub>h <sub>as</sub> CLIP<sub>-</sub>b<sub>ase</sub>d decodin<sub>g</sub> [35] and self-su<sub>p</sub>ervised visual re<sub>p</sub>resentations includin<sub>g</sub> DINO-st<sub>y</sub>le encoders [7]. Thou<sub>g</sub>h dominated b<sub>y</sub> fMRI data, [1, 12, 32, 33, 38, 73] ex<sub>p</sub>lored on EEG data, obtained <sub>p</sub>romisin<sub>g</sub> results.

To further modulate activations, com<sub>p</sub>utational models [47, 59] and frameworks tar<sub>g</sub>etin<sub>g</sub> sin<sub>g</sub>le neuron contributions [5, 55] <sub>p</sub>rovide inter<sub>p</sub>retable ma<sub>pp</sub>in<sub>g</sub>s of cortical activations to s<sub>p</sub>ecific visual features. Latent s<sub>p</sub>ace o<sub>p</sub>timization strate<sub>g</sub>ies [66, 82] extend foundational decodin<sub>g</sub> frameworks [25, 48, 53, 71] to achieve tar<sub>g</sub>eted neural activation modulation. BrainDiVE and BrainSCUBA[36, 37] <sub>u</sub>ses a CLIP-based encoder and dif<sub>u</sub>sion models to im<sub>p</sub>ro<sub>v</sub>e stim<sub>u</sub>- lus <sub>q</sub>ualit<sub>y</sub> and semantic s<sub>p</sub>ecificit<sub>y</sub>. BrainACTIV [17], as the recent <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t <sub>an</sub>d th<sub>e</sub> fi<sub>rs</sub>t <sub>wor</sub>k <sub>on us</sub>i<sub>ng</sub> dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l<sub>s</sub> t<sub>o gen-</sub> erate ima<sub>g</sub>es to re<sub>g</sub>ulate brain activations<sub>,</sub> extends these <sub>p</sub>aradi<sub>g</sub>ms b<sub>y</sub> conditionin<sub>g</sub> s<sub>y</sub>nthesis with linearl<sub>y</sub> fitted CLIP embeddin<sub>g</sub>s <sub>an</sub>d b<sub>ra</sub>i<sub>n ac</sub>ti<sub>va</sub>ti<sub>on pa</sub>tt<sub>erns.</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d <sub>e</sub>f<sub>or</sub>t<sub>s</sub> i<sub>n con</sub>t<sub>ro</sub>ll<sub>a</sub>bl<sub>e ac</sub>ti<sub>-</sub> <sub>va</sub>ti<sub>on syn</sub>th<sub>es</sub>i<sub>s</sub> h<sub>ave a</sub>l<sub>so s</sub>h<sub>own</sub> th<sub>a</sub>t <sub>genera</sub>ti<sub>ve mo</sub>d<sub>e</sub>l<sub>s can</sub> b<sub>e</sub> used not onl<sub>y</sub> to maximize neural res<sub>p</sub>onses but also to su<sub>pp</sub>ress <sub>or</sub> dif<sub>eren</sub>ti<sub>a</sub>ll<sub>y mo</sub>d<sub>u</sub>l<sub>a</sub>t<sub>e</sub> th<sub>em,</sub> b<sub>o</sub>th i<sub>n</sub> bi<sub>o</sub>l<sub>og</sub>i<sub>ca</sub>l <sub>sys</sub>t<sub>ems an</sub>d in artificial networks [4, 49, 72]. Althou<sub>g</sub>h modelin<sub>g</sub> activations via CLIP embeddin<sub>g</sub>s does work in <sub>p</sub>revious work<sub>,</sub> our method tries to take the nois<sub>y</sub> and mix-selective nature of brain activit<sub>y</sub> d<sub>a</sub>t<sub>a</sub> i<sub>n</sub>t<sub>o</sub> <sub>cons</sub>id<sub>era</sub>ti<sub>on,</sub> b<sub>u</sub>ildi<sub>ng</sub> <sub>a</sub> <sub>nove</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>an</sub>d<sub>scape</sub> i<sub>n</sub> VAE throu<sub>g</sub>h CLIP embeddin<sub>g</sub>s<sub>, p</sub>ermittin<sub>g</sub> more controllable and efective activation mani<sub>p</sub>ulation<sub>,</sub> and avoidin<sub>g</sub> fallin<sub>g</sub> into local minim<sub>u</sub>m in th<sub>e op</sub>timiz<sub>a</sub>ti<sub>o</sub>n <sub>p</sub>r<sub>ocess.</sub>

## 3 Methodology

## 3.1 Problem Setting and Notation

Let I be the s<sub>p</sub>ace of natural ima<sub>g</sub>es, X be the s<sub>p</sub>ace of CLIP emb<sub>e</sub>ddi<sub>ngs, an</sub>d $z$ b<sub>e</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>space o</sub>f <sub>a var</sub>i<sub>a</sub>ti<sub>ona</sub>l <sub>au</sub>t<sub>o-enco</sub>d<sub>er</sub> (VAE). For an in<sub>p</sub>ut ima<sub>g</sub>e $i \in \mathcal { I } ,$ , the <sub>p</sub>retrained CLIP encoder E ma<sub>p</sub>s � to a CLIP embeddin<sub>g</sub> $x = \mathcal { E } ( i ) \in \mathcal { X }$ . S<sub>u</sub>bse<sub>qu</sub>entl<sub>y,</sub> the VAE <sub>enco</sub>d<sub>er</sub> $q _ { \phi } ( z | x )$ <sub>pro</sub>d<sub>uces</sub> <sub>a</sub> l<sub>a</sub>t<sub>en</sub>t <sub>�</sub> $\in \mathcal { Z }$ <sub>w</sub>ith <sub>mean � an</sub>d di<sub>agona</sub>l <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> $\Sigma = \mathrm { d i a g } ( \sigma ^ { 2 } )$ <sub>;</sub> th<sub>e</sub> VAE d<sub>eco</sub>d<sub>er</sub> $p _ { \theta } ( x \mid z )$ reconstructs <sub>an</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> $x _ { \mathrm { r e c } } .$ Th<sub>roug</sub>h<sub>ou</sub>t<sub>, �</sub> d<sub>eno</sub>t<sub>es a sca</sub>l<sub>ar ac</sub>ti<sub>va</sub>ti<sub>on va</sub>l<sub>ue</sub> <sub>o</sub>f <sub>a c</sub>h<sub>osen</sub> t<sub>arge</sub>t<sub>,</sub> i<sub>.e.,</sub> th<sub>e mean response o</sub>f <sub>a se</sub>l<sub>ec</sub>t<sub>e</sub>d ROI i<sub>n</sub> h<sub>u-</sub> <sub>man</sub> b<sub>ra</sub>i<sub>n</sub> d<sub>a</sub>t<sub>a a</sub>ft<sub>er z-score norma</sub>li<sub>za</sub>ti<sub>on or</sub> th<sub>e response o</sub>f <sub>a</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>un</sub>it i<sub>n an ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>.</sub> $\lambda , \omega _ { \mathrm { r e c } } .$ <sub>�KLD are</sub> <sub>sca</sub>l<sub>ar</sub> <sup>h</sup>yp<sup>er-</sup>p<sup>arameters.</sup>

## 3.2 Constructing a Structured VAE Landscape with Activation-Constrained Priors

T<sub>o genera</sub>t<sub>e exp</sub>l<sub>a</sub>i<sub>na</sub>bl<sub>e an</sub>d <sub>con</sub>t<sub>ro</sub>ll<sub>a</sub>bl<sub>e pre</sub>f<sub>erre</sub>d <sub>v</sub>i<sub>sua</sub>l <sub>represen-</sub> tations for a s<sub>p</sub>ecific brain re<sub>g</sub>ion or neural unit<sub>,</sub> we incor<sub>p</sub>orate th<sub>e</sub> t<sub>a</sub>r<sub>ge</sub>t <sub>ac</sub>ti<sub>va</sub>ti<sub>o</sub>n <sub>�</sub> int<sub>o</sub> th<sub>e</sub> VAE <sub>p</sub>ri<sub>o</sub>r <sub>a</sub>nd <sub>use</sub> it t<sub>o o</sub>r<sub>ga</sub>niz<sub>e</sub> th<sub>e</sub> latent s<sub>p</sub>ace. The ke<sub>y</sub> intuition is that neural si<sub>g</sub>nals are nois<sub>y</sub> and <sub>cou</sub>ld h<sub>ave an asymme</sub>t<sub>r</sub>i<sub>c</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> T<sub>o</sub> b<sub>e</sub>tt<sub>er represen</sub>t th<sub>ose</sub> <sub>p</sub>ro<sub>p</sub>erties<sub>,</sub> we desi<sub>g</sub>n a <sub>p</sub>rior for the neural res<sub>p</sub>onses that the<sub>y</sub> have a bimodal distribution, with si<sub>g</sub>nificant res<sub>p</sub>onses (most <sub>p</sub>ositive or ne<sub>g</sub>ative) are ex<sub>p</sub>ected to be centralized in each distribution. The <sub>grea</sub>t<sub>er</sub> th<sub>e</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f <sub>ac</sub>ti<sub>va</sub>ti<sub>on</sub> i<sub>s,</sub> th<sub>e</sub> <sub>c</sub>l<sub>oser</sub> it <sub>w</sub>ill b<sub>e</sub> t<sub>o</sub> <sub>eac</sub>h di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on cen</sub>t<sub>er, an</sub>d <sub>responses c</sub>l<sub>ose</sub> t<sub>o zero are re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> l<sub>oose</sub>l<sub>y</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>ne</sub>d<sub>.</sub> Thi<sub>s</sub> d<sub>es</sub>i<sub>gn</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>a</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>a</sub>t<sub>en</sub>t l<sub>an</sub>d<sub>scape</sub> that su<sub>pp</sub>orts direct latent mani<sub>p</sub>ulation at test time.

The KL diver<sub>g</sub>ence (KLD) term in the VAE measures the diver-<sub>g</sub>ence between the encoder’s a<sub>pp</sub>roximate <sub>p</sub>osterior<sub>,</sub> $q _ { \phi } ( \mathbf { z } | \mathbf { x } )$ <sub>,</sub> <sub>an</sub>d the <sub>p</sub>rior<sub>,</sub> $p _ { \boldsymbol { \theta } } ( \mathbf { z } )$ <sub>.</sub> W<sub>e</sub> b<sub>eg</sub>i<sub>n</sub> <sub>w</sub>ith it<sub>s</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>e</sub>fi<sub>n</sub>iti<sub>on:</sub>

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathrm { K L } } ( q _ { \phi } ( { \bf z } | { \bf x } ) \parallel \phi _ { \theta } ( { \bf z } ) ) = \int q _ { \phi } ( { \bf z } | { \bf x } ) \log \frac { q _ { \phi } ( { \bf z } | { \bf x } ) } { \phi _ { \theta } ( { \bf z } ) } d { \bf z } } \ ~ } \\ { { \displaystyle ~ = \mathbb { E } _ { { \bf z } \sim q _ { \phi } ( { \bf z } | { \bf x } ) } \left[ \log q _ { \phi } ( { \bf z } | { \bf x } ) - \log p _ { \theta } ( { \bf z } ) \right] . } } \end{array}\tag{1}
$$

Both <sub>p</sub>osterior and <sub>p</sub>rior are defined as multivariate Gaussian distrib<sub>u</sub>tions with dia<sub>g</sub>onal covariance matrices. In o<sub>u</sub>r framework<sub>,</sub> the a<sub>pp</sub>roximate <sub>p</sub>osterior is $q _ { \phi } ( \mathbf { z } \mid \mathbf { x } ) = N ( \mathbf { z } ; \pmb { \mu } , \mathrm { d i a g } ( \pmb { \sigma } ^ { 2 } ) )$ ). For the prior, we de<sup>fi</sup>ne an activation-constrained prior:

$$
\begin{array} { r } { p _ { \theta } ( { \bf z } ) = N ( { \bf z } ; \mu ^ { \mathrm { p r i o r } } , \mathrm { d i a g } ( ( \sigma ^ { \mathrm { p r i o r } } ) ^ { 2 } ) ) , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h d<sub>epen</sub>d<sub>s on</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>ac</sub>ti<sub>va</sub>ti<sub>on va</sub>l<sub>ue �:</sub>

$$
\sigma _ { i } ^ { \mathrm { p r i o r } } = \Bigg | \frac { \lambda } { r _ { i } + \epsilon } \Bigg | , \qquad \mu _ { i } ^ { \mathrm { p r i o r } } = \Bigg \{ \mu _ { \mathrm { p o s } } , \quad \mathrm { i f } r > 0 ,\tag{2}
$$

Here<sub>,</sub> � is a scalin<sub>g</sub> h<sub>yp</sub>er<sub>p</sub>arameter<sub>,</sub> <sub>�</sub> is a small constant to <sub>p</sub>revent division b<sub>y</sub> zero<sub>,</sub> and $\mu _ { \mathrm { p o s } } , \mu _ { \mathrm { n e g } }$ <sub>are</sub> <sub>pre</sub>d<sub>e</sub>fi<sub>ne</sub>d <sub>means</sub> f<sub>or</sub> <sub>p</sub>ositive and ne<sub>g</sub>ative activations<sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>. The role of $\mu _ { \mathrm { p o s } }$ <sub>an</sub>d <sub>�</sub> is to <sub>p</sub>rovide two se<sub>p</sub>arated latent centers in VAE latent s<sub>p</sub>ace f<sub>o</sub>r m<sub>a</sub>ximiz<sub>e ac</sub>ti<sub>va</sub>ti<sub>o</sub>n in<sub>c</sub>r<sub>ease a</sub>nd d<sub>ec</sub>r<sub>ease.</sub> M<sub>ea</sub>n<sub>w</sub>hil<sub>e, va</sub>ri<sub>a</sub>n<sub>ce</sub> term $\sigma _ { i } ^ { \mathrm { p r i o r } }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h d<sub>e</sub>fi<sub>nes</sub> dif<sub>eren</sub>t <sub>pr</sub>i<sub>ors</sub> f<sub>or</sub> <sub>eac</sub>h d<sub>a</sub>t<sub>a</sub> <sub>po</sub>i<sub>n</sub>t �<sub>,</sub> d<sub>e-</sub> crease as |�<sub>�</sub> | increases. B<sub>y</sub> im<sub>p</sub>osin<sub>g</sub> such continuous loose to ti<sub>g</sub>ht constraints for each data <sub>p</sub>oint<sub>,</sub> our method can ma<sub>p</sub> ima<sub>g</sub>es to a <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>space.</sub> Th<sub>eore</sub>ti<sub>ca</sub>ll<sub>y,</sub> th<sub>e</sub> i<sub>npu</sub>t<sub>s</sub> <sub>w</sub>ill b<sub>e</sub> <sub>cen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>w</sub>h<sub>en</sub> $r _ { i }$ a<sub>pp</sub>roaches infinit<sub>y,</sub> while in <sub>p</sub>ractice<sub>,</sub> we can sam<sub>p</sub>le in<sub>p</sub>ut data <sub>p</sub>oints with maximize and minimize values at two <sub>p</sub>re-defined cen t<sub>ers,</sub> <sub>an</sub>d th<sub>ey</sub> <sub>are</sub> <sub>expec</sub>t<sub>e</sub>d t<sub>o</sub> i<sub>ncrease</sub> <sub>or</sub> d<sub>ecrease</sub> th<sub>e</sub> <sub>ac</sub>ti<sub>va</sub>ti<sub>ons</sub> <sub>o</sub>f t<sub>arge</sub>t<sub>e</sub>d <sub>neurons.</sub>

![](images/8370e24995d043e5aff526dd6f825a1c171c17aacd9c704daa9d8c0812a543c0.jpg)  
Figure 2: Overview of Vis-Lens. The illustration depicts a two-stage approach integrating a structured landscape and variational autoencoder (VAE) generation with CLIP encoder and difusion based generative model. Phase I: The input image, $i _ { \mathbf { t r a i n } } ,$ is encoded by a pretrained CLIP encoder $\varepsilon ,$ CLIP embeddings � are encoded into a latent representation � by the VAE encoder. A regularization step using a activation-based prior $ { N } ( \mu _ { \mathrm { p r i o r } } , \frac {  { 1 } } { | r | } )$ modifies the latent space for better alignment with activations, producing a regularized latent $z .$ The VAE is also trained to reconstruct the input CLIP embeddings � with $x _ { r e c o n } .$ . Phase II: The test images $i _ { \mathrm { t e s t } } ,$ is encoded by same CLIP encoder E. Then, with fully trained VAE and the structured visual latent representation space, we can directly modify latents constrained by a specific feature represented by the selected regions, obtaining a new latent $z ^ { \prime } .$ . This transformed latent is decoded back into an intermediate generated CLIP embeddings $x _ { g e n } ,$ which is used to guild the image generation through a difusion pipeline, yielding the final output $i _ { \mathbf { g e n } }$ as preferred images to modulate the activation $r _ { \ast }$

Given the definition of the a<sub>pp</sub>roximate <sub>p</sub>osterior and activationconstrained <sub>p</sub>rior<sub>,</sub> the ex<sub>p</sub>ectation in E<sub>q</sub>. 1 can be com<sub>p</sub>uted anal<sub>y</sub>ticall<sub>y,</sub> <sub>y</sub>ieldin<sub>g</sub> the closed-form KLD between two Gaussian di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub> A<sub>ssu</sub>min<sub>g</sub> th<sub>a</sub>t th<sub>e</sub> VAE l<sub>a</sub>t<sub>e</sub>nt <sub>space</sub> h<sub>as</sub> $d _ { z }$ di<sub>men-</sub> <sub>s</sub>i<sub>o</sub>n<sub>s,</sub> th<sub>e</sub> fin<sub>a</sub>l <sub>ac</sub>ti<sub>va</sub>ti<sub>o</sub>n-<sub>awa</sub>r<sub>e</sub> KLD i<sub>s</sub>

$$
\mathrm { K L D } _ { \mathrm { a c t i v a t i o n } } = \sum _ { i = 1 } ^ { d _ { z } } \left[ \log \frac { \sigma _ { i } ^ { \mathrm { p r i o r } } } { \sigma _ { i } } - \frac { 1 } { 2 } + \frac { \sigma _ { i } ^ { 2 } + ( \mu _ { i } - \mu _ { i } ^ { \mathrm { p r i o r } } ) ^ { 2 } } { 2 ( \sigma _ { i } ^ { \mathrm { p r i o r } } ) ^ { 2 } } \right] .\tag{3}
$$

Here, $\mu _ { i }$ <sub>an</sub>d $\sigma _ { i }$ <sub>are</sub> th<sub>e</sub> <sub>mean</sub> <sub>an</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>ap-</sub> <sub>p</sub>roximate <sub>p</sub>osterior for the �-th latent dimension<sub>,</sub> res<sub>p</sub>ectivel<sub>y;</sub> $\mu _ { i } ^ { \mathrm { p r i o r } }$ <sub>an</sub>d $\sigma _ { i } ^ { \mathrm { p r i o r } }$ <sub>are</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>mean</sub> <sub>an</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>pr</sub>i<sub>or.</sub> Thi<sub>s</sub> <sub>mo</sub>difi<sub>e</sub>d KLD d<sub>oes</sub> <sub>more</sub> th<sub>an</sub> <sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t s<sub>p</sub>ace toward a standard Gaussian: it ex<sub>p</sub>licitl<sub>y</sub> or<sub>g</sub>anizes latent <sub>co</sub>d<sub>es accor</sub>di<sub>ng</sub> t<sub>o ac</sub>ti<sub>va</sub>ti<sub>ons.</sub> A<sub>s a resu</sub>lt<sub>,</sub> th<sub>e</sub> l<sub>earne</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>space</sub> b<sub>ecomes more su</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or con</sub>t<sub>ro</sub>ll<sub>a</sub>bl<sub>e ac</sub>ti<sub>va</sub>ti<sub>on mo</sub>d<sub>u</sub>l<sub>a</sub>ti<sub>on</sub> th<sub>an</sub> a sin<sub>g</sub>le linear direction in the ori<sub>g</sub>inal embeddin<sub>g</sub> s<sub>p</sub>ace.

With this acti<sub>v</sub>ation-constrained <sub>p</sub>rior<sub>,</sub> <sub>w</sub>e can b<sub>u</sub>ild a str<sub>u</sub>ct<sub>u</sub>red VAE latent s<sub>p</sub>ace<sub>,</sub> where we can easil<sub>y</sub> <sub>p</sub>erform in-distribution sam-<sub>p</sub>lin<sub>g</sub> from a s<sub>p</sub>herical s<sub>p</sub>ace in a structured landsca<sub>p</sub>e.

## 3.3 Generate Preferred Visual Representations with Modified VAE

T<sub>o</sub> <sub>pro</sub>d<sub>uce</sub> <sub>pre</sub>f<sub>erre</sub>d i<sub>mages,</sub> <sub>our</sub> <sub>me</sub>th<sub>o</sub>d <sub>cons</sub>i<sub>s</sub>t<sub>s</sub> <sub>o</sub>f t<sub>wo</sub> <sub>p</sub>h<sub>ases:</sub> <sub>a</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng p</sub>h<sub>ase</sub> th<sub>a</sub>t <sub>cons</sub>t<sub>ruc</sub>t<sub>s</sub> th<sub>e s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>space, an</sub>d a <sub>g</sub>eneration <sub>p</sub>hase that s<sub>y</sub>nthesizes ima<sub>g</sub>es from it. The VAE is

Structured Visual Landscape: Generating Preferred Representations in Biological and Artificial Neural Networks

![](images/9cd7b6c26b385392e8262ca6758eaaab2aa536e46882656974a5dce6f7c747ab.jpg)  
Figure 3: Activation change results of our method and BrainACTIV [17] (A) Generate preferred representations of multiple ROIs with example images from COCO dataset by our method and baselines. The alpha value in interpolation indicates the strength and direction of preferred semantic in generation. (B) Comparison between predicted activations of ROIs made by DINO-ViT encoder. Means and ranges of activations are presented for diferent alpha settings. Our method outperforms the baseline in terms of maximizing or minimizing the ROI activations.

the onl<sub>y</sub> trainable com<sub>p</sub>onent in the <sub>p</sub>i<sub>p</sub>eline<sub>,</sub> while the <sub>p</sub>retrained CLIP encoder (E) and the ima<sub>g</sub>e-to-ima<sub>g</sub>e difusion model remain f<sub>rozen.</sub>

Training. During training, on<sup>l</sup>y t<sup>h</sup>e VAE encoder $q _ { \phi } ( z \mid x )$ <sub>an</sub>d d<sub>eco</sub>d<sub>er</sub> $p _ { \theta } ( x | z )$ <sub>,</sub> which ma<sub>p</sub> the CLIP embeddin<sub>g</sub> to a latent re<sub>p</sub>re-<sub>sen</sub>t<sub>a</sub>ti<sub>on � an</sub>d <sub>recons</sub>t<sub>ruc</sub>t it b<sub>ac</sub>k t<sub>o</sub> $x _ { \mathrm { r e c o n } }$ <sub>,</sub> are o<sub>p</sub>timized b<sub>y</sub>

$$
\mathcal { L } _ { \mathrm { V A E } } = \omega _ { \mathrm { r e c o n } } \cdot \mathcal { L } _ { \mathrm { r e c o n } } + \omega _ { \mathrm { K L D } } \cdot \mathrm { K L D } _ { \mathrm { a c t i v a t i o n } } ,\tag{4}
$$

<sub>w</sub>h<sub>ere</sub> $\mathcal { L } _ { \mathrm { r e c o n } } = \| x - x _ { \mathrm { r e c o n } } \| _ { 2 } ^ { 2 }$ i<sub>s</sub> th<sub>e recons</sub>t<sub>ruc</sub>ti<sub>on</sub> l<sub>oss</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>an</sub>d d<sub>eco</sub>d<sub>e</sub>d CLIP <sub>em</sub>b<sub>e</sub>ddi<sub>ngs.</sub> Th<sub>e</sub> t<sub>erms</sub> $\omega _ { \mathrm { r e c o n } }$ <sub>an</sub>d <sub>�KLD</sub> b<sub>a</sub>l<sub>ance recons</sub>t<sub>ruc</sub>ti<sub>on</sub> fid<sub>e</sub>lit<sub>y aga</sub>i<sub>ns</sub>t th<sub>e ac</sub>ti<sub>va</sub>ti<sub>on-</sub>b<sub>ase</sub>d re<sub>g</sub>ularization im<sub>p</sub>osed b<sub>y</sub> E<sub>q</sub>. 3. Therefore<sub>,</sub> the VAE is trained not <sub>on</sub>l<sub>y</sub> t<sub>o recons</sub>t<sub>ruc</sub>t CLIP <sub>em</sub>b<sub>e</sub>ddi<sub>ngs,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> t<sub>o res</sub>h<sub>ape</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t s<sub>p</sub>ace such that neural activation information becomes <sub>g</sub>eometri-<sub>ca</sub>ll<sub>y access</sub>ibl<sub>e</sub> f<sub>or</sub> l<sub>a</sub>t<sub>er man</sub>i<sub>pu</sub>l<sub>a</sub>ti<sub>on.</sub>

Generation. To synt<sup>h</sup>esize new images, we <sup>fi</sup>rst encode an input im<sub>age</sub> � <sub>w</sub>ith th<sub>e</sub> fr<sub>o</sub>z<sub>e</sub>n CLIP <sub>e</sub>n<sub>co</sub>d<sub>e</sub>r t<sub>o o</sub>bt<sub>a</sub>in it<sub>s</sub> CLIP <sub>e</sub>mb<sub>e</sub>ddin<sub>g</sub> $x = \mathcal { E } ( i )$ <sub>.</sub> W<sub>e</sub> th<sub>en</sub> f<sub>ee</sub>d <sub>�</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> t<sub>ra</sub>i<sub>ne</sub>d VAE <sub>enco</sub>d<sub>er</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> <sub>a</sub> l<sub>a</sub>t<sub>en</sub>t <sub>co</sub>d<sub>e</sub> $z = q _ { \phi } ( x )$ i<sub>n</sub> th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>a</sub>t<sub>en</sub>t l<sub>an</sub>d<sub>scape.</sub> N<sub>ex</sub>t<sub>,</sub> <sub>�</sub> is moved toward the <sub>p</sub>redefined <sub>p</sub>ositive or ne<sub>g</sub>ative <sub>p</sub>rior center<sub>,</sub> <sub>pro</sub>d<sub>uc</sub>i<sub>ng</sub> <sub>a</sub> <sub>mo</sub>difi<sub>e</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>co</sub>d<sub>e</sub> $z ^ { \prime } .$ <sub>.</sub> Th<sub>e</sub> <sub>mo</sub>difi<sub>e</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>co</sub>d<sub>e</sub> i<sub>s</sub> decoded into a new CLIP embeddin<sub>g</sub> $x ^ { \prime } = p _ { \theta } ( x \mid z ^ { \prime } )$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> th<sub>en</sub> <sub>passe</sub>d t<sub>o</sub> th<sub>e</sub> f<sub>rozen</sub> dif<sub>us</sub>i<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l t<sub>o</sub> <sub>syn</sub>th<sub>es</sub>i<sub>ze</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>pre</sub>f<sub>erre</sub>d im<sub>age</sub> �<sup>′</sup><sub>.</sub>

## 4 Experiments

## 4.1 Datasets

Natural Scenes Dataset (NSD). T<sup>h</sup>e Natura<sup>l</sup> Scenes Dataset (NSD), introduced b<sub>y</sub> Allen et al. [2], <sub>p</sub>rovides hi<sub>g</sub>h-resolution fMRI recordings <sup>f</sup>rom eig<sup>h</sup>t su<sup>b</sup>jects viewing approximate<sup>l</sup>y 10,000 MS COCO ima<sub>g</sub>es [34], each re<sub>p</sub>eated three times to enhance data consistenc<sub>y</sub>. Followin<sub>g</sub> standard fMRI encodin<sub>g p</sub>ractices [48, 78], beta values <sub>a</sub>r<sub>e</sub> z-<sub>sco</sub>r<sub>e</sub>d <sub>w</sub>ithin <sub>sess</sub>i<sub>o</sub>n<sub>s a</sub>nd <sub>ave</sub>r<sub>age</sub>d <sub>ac</sub>r<sub>oss</sub> r<sub>epe</sub>titi<sub>o</sub>n<sub>s.</sub> ROI masks are utilized to extract activation <sub>p</sub>atterns from s<sub>p</sub>ecific brain re<sub>g</sub>ions<sub>,</sub> servin<sub>g</sub> as essential constraints in our framework to ali<sub>g</sub>n th<sub>e</sub> VAE l<sub>a</sub>t<sub>e</sub>nt <sub>space</sub> <sub>w</sub>ith <sub>o</sub>b<sub>se</sub>r<sub>ve</sub>d br<sub>a</sub>in r<sub>espo</sub>n<sub>ses</sub> in 4<sub>.</sub>3<sub>.</sub>1<sub>.</sub>

ImageNet-mini. ImageNet-mini is a subset o<sup>f</sup> t<sup>h</sup>e <sup>f</sup>u<sup>ll</sup> ImageNet dataset [13], oferin<sub>g</sub> a diverse collection of ima<sub>g</sub>es across various <sub>ca</sub>t<sub>egor</sub>i<sub>es.</sub> Thi<sub>s</sub> d<sub>a</sub>t<sub>ase</sub>t <sub>serves</sub> <sub>as</sub> <sub>a</sub> b<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>p</sub>erformance of modulatin<sub>g</sub> artificial units from SIMCLR [8] in 4.3.2 and validatin<sub>g</sub> o<sub>p</sub>timization <sub>p</sub>erformance under constrained ima<sub>g</sub>e dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n<sub>s</sub> in 4<sub>.</sub>3<sub>.</sub>3<sub>.</sub>

Table 1: Comparison of activation change and realism between our method and the baseline on NSD datasets. Better values are in bold face. For activation changes, ↑ indicates that higher values are better, and ↓ indicates that lower values are better. Our method produces activation changes that are more consistently aligned with the intended direction.  
(a) NSD Results
<table><tr><td></td><td>ROIs</td><td>TE2p</td><td>IPO</td><td>IP1</td><td>OFA</td><td>FFA</td><td>EBA</td><td>OPA</td></tr><tr><td rowspan="2">Activation Increase (↑)</td><td>Our Method</td><td>0.242</td><td>0.498</td><td>0.153</td><td>1.052</td><td>0.891</td><td>0.963</td><td>0.699</td></tr><tr><td>BrainACTIV</td><td>0.207</td><td>0.463</td><td>0.044</td><td>1.155</td><td>0.976</td><td>1.160</td><td>0.699</td></tr><tr><td rowspan="2">Activation Decrease (↓)</td><td>Our Method</td><td>-0.136</td><td>-0.699</td><td>-0.073</td><td>-0.660</td><td>-0.580</td><td>-0.810</td><td>-0.824</td></tr><tr><td>BrainACTIV</td><td>0.149</td><td>0.186</td><td>0.092</td><td>0.213</td><td>-0.484</td><td>-0.669</td><td>-0.172</td></tr><tr><td rowspan="2">FID @ 2k (↓)</td><td>Our Method</td><td>172.71</td><td>173.34</td><td>178.01</td><td>173.66</td><td>153.70</td><td>137.33</td><td>166.01</td></tr><tr><td> $\mathrm { B r a i n A C T I V }$ </td><td>225.77</td><td>213.17</td><td>200.14</td><td>216.15</td><td>256.44</td><td>188.59</td><td>225.37</td></tr></table>

Table 2: Cross-subject generalization results on the NSD dataset. The table compares predicted activation changes obtained from the original subject-specific predictor and a held-out-subject predictor across several visual ROIs.
<table><tr><td>ROIs</td><td>OFA</td><td>FFA</td><td>EBA</td><td>VWFA</td><td>OPA</td><td>PPA</td><td>RSC</td></tr><tr><td>Activation Increase (original subj.)</td><td>0.869</td><td>0.851</td><td>0.860</td><td>0.867</td><td>0.731</td><td>0.966</td><td>0.942</td></tr><tr><td>Activation Increase (held-out subj.)</td><td>0.861</td><td>0.975</td><td>0.790</td><td>0.648</td><td>0.629</td><td>1.061</td><td>0.940</td></tr><tr><td>Activation Decrease (original subj.)</td><td>-0.526</td><td>-0.382</td><td>-0.564</td><td>-0.279</td><td>-0.557</td><td>-0.610</td><td>-0.775</td></tr><tr><td>Activation Decrease (held-out subj.)</td><td>-0.436</td><td>-0.302</td><td>-0.391</td><td>-0.089</td><td>-0.485</td><td>-0.813</td><td>-0.823</td></tr></table>

## 4.2 Models

Implementation of our framework. Our VAE uses a <sup>l</sup>inear encoder to ma<sub>p</sub> CLIP embeddin<sub>g</sub>s into an acti<sub>v</sub>ation-constrained latent s<sub>p</sub>ace and a linear decoder to reconstruct CLIP embeddin<sub>g</sub>s f<sub>o</sub>r <sub>co</sub>ntr<sub>o</sub>ll<sub>e</sub>d im<sub>age</sub> <sub>sy</sub>nth<sub>es</sub>i<sub>s.</sub> W<sub>e</sub> <sub>use</sub> <sub>a</sub> <sub>p</sub>r<sub>e</sub>tr<sub>a</sub>in<sub>e</sub>d CLIP-ViT-H-14 encoder [64] to extract 1024-dimensional embeddin<sub>g</sub>s, and Stable Difusion v1.5 with IP-Ada<sub>p</sub>ter [83] as the ima<sub>g</sub>e <sub>g</sub>enerator.

Baseline method. For every ROI, we re-implement BrainACTIV [17], a ridge-regularized linear model on ℓ<sub>2</sub>-normalized CLIP embeddings, $\left[ \begin{array} { l } { { \frac { x } { \| x \| } } \mathbf { \cdot w } + b } \end{array} \right] \implies r ,$ yielding a single modulation vector w $\in \mathbb { R } ^ { 1 0 \bar { 2 } 4 }$ Thi<sub>s vec</sub>t<sub>o</sub>r i<sub>s</sub> int<sub>e</sub>r<sub>p</sub>r<sub>e</sub>t<sub>e</sub>d <sub>as</sub> th<sub>e</sub> dir<sub>ec</sub>ti<sub>o</sub>n <sub>o</sub>f m<sub>a</sub>xim<sub>a</sub>l <sub>a</sub>nd minim<sub>a</sub>l <sub>ac</sub>ti<sub>va</sub>ti<sub>o</sub>n in CLIP <sub>space</sub>: $\mathbf { z } _ { \mathrm { m a x } } = \mathbf { w } / \| \mathbf { w } \|$ <sub>an</sub>d ${ \bf z } _ { \mathrm { m i n } } = - { \bf w } / \| { \bf w } \| .$ <sub>.</sub> At inference time, a test CLIP embedding is shifted only along z<sub>max</sub> or $\mathbf { z } _ { \mathrm { m i n } }$ <sup>b</sup>e<sup>f</sup>ore <sup>b</sup>eing injecte<sup>d</sup>, via an unc<sup>h</sup>ange<sup>d</sup> a<sup>d</sup>apter, into a <sup>f</sup>rozen i<sub>mage-</sub>t<sub>o-</sub>i<sub>mage</sub> dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l<sub>.</sub> Thi<sub>s</sub> b<sub>ase</sub>li<sub>ne</sub> th<sub>ere</sub>f<sub>ore assumes</sub> that activation increase and activation decrease can be re<sub>p</sub>resented b<sub>y</sub> a sin<sub>g</sub>le s<sub>y</sub>mmetric direction in CLIP s<sub>p</sub>ace<sub>,</sub> whereas our method <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> th<sub>em</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y.</sub>

In-silico brain simulator. In our experiments, brain activations are evaluated with a DINO-ViT encoder [57]. DINO acts as a frozen f<sub>ea</sub>t<sub>ure ex</sub>t<sub>rac</sub>t<sub>or w</sub>h<sub>ose</sub> 12 l<sub>ayer ou</sub>t<sub>pu</sub>t<sub>s are</sub> f<sub>e</sub>d t<sub>o a s</sub>i<sub>ng</sub>l<sub>e-</sub>l<sub>ayer</sub> ViT ensemble trained with rid<sub>g</sub>e loss to <sub>p</sub>redict neuronal res<sub>p</sub>onses. In line with the findin<sub>g</sub>s of [17], this encoder ex<sub>p</sub>lains a lar<sub>g</sub>e <sub>p</sub>ro-<sub>p</sub>ortion of NSD voxel variance (t<sub>yp</sub>icall<sub>y</sub> $R ^ { 2 } > 0 . 6$ i<sub>n</sub> hi<sub>g</sub>h<sub>-</sub>l<sub>eve</sub>l ROIs), validatin<sub>g</sub> its use as a reliable neural readout. Because the encoder is not CLIP-based<sub>,</sub> it <sub>p</sub>ro<sub>v</sub>ides an inde<sub>p</sub>endent estimate of ne<sub>u</sub>ral acti<sub>v</sub>it<sub>y</sub>.

## 4.3 Results

## 4.3.1 Generate Preferred Representations in Human Brain.

With th<sub>e s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>an</sub>d<sub>scape, we can o</sub>bt<sub>a</sub>i<sub>n a</sub> b<sub>e</sub>tt<sub>er represen</sub>t<sub>a-</sub> tion of a tar<sub>g</sub>et brain re<sub>g</sub>ion b<sub>y</sub> sam<sub>p</sub>lin<sub>g</sub> around the corres<sub>p</sub>ondin<sub>g</sub> <sub>p</sub>rior in latent s<sub>p</sub>ace and <sub>g</sub>eneratin<sub>g</sub> ima<sub>g</sub>es from those latent values<sub>,</sub> <sub>as</sub> <sub>s</sub>h<sub>ow</sub>n in Fi<sub>gu</sub>r<sub>e</sub> 1<sub>.</sub>

I<sub>n our</sub> l<sub>a</sub>t<sub>en</sub>t <sub>space, we</sub> i<sub>n</sub>t<sub>erpo</sub>l<sub>a</sub>t<sub>e eac</sub>h l<sub>a</sub>t<sub>en</sub>t <sub>co</sub>d<sub>e �</sub> t<sub>owar</sub>d <sub>a</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>o</sub>r n<sub>ega</sub>ti<sub>ve</sub> <sub>p</sub>ri<sub>o</sub>r<sub>,</sub> $z ^ { \prime } = \alpha \cdot \mu _ { \mathrm { p r i o r } } + ( 1 - \alpha ) \cdot z$ <sub>,</sub> and <sub>p</sub>ass it t<sub>o a</sub> f<sub>rozen</sub> SDEdit dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l <sub>w</sub>ith $\gamma = 1 . 0 ;$ th<sub>us,</sub> th<sub>e ou</sub>t<sub>pu</sub>t i<sub>mage</sub> d<sub>epen</sub>d<sub>s so</sub>l<sub>e</sub>l<sub>y on</sub> th<sub>e e</sub>dit<sub>e</sub>d <sub>em</sub>b<sub>e</sub>ddi<sub>ng, a</sub>ll<sub>ow</sub>i<sub>ng a c</sub>l<sub>ean</sub> <sub>compar</sub>i<sub>son w</sub>ith th<sub>e</sub> b<sub>ase</sub>li<sub>ne.</sub>

Fi<sub>g</sub>ure 3 and Table 1 show results on the NSD dataset. Our <sub>me</sub>th<sub>o</sub>d <sub>pro</sub>d<sub>uces ac</sub>ti<sub>va</sub>ti<sub>on c</sub>h<sub>anges</sub> i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d di<sub>rec</sub>ti<sub>on</sub> across all re<sub>p</sub>orted ROIs, whereas the BrainACTIV [17] fails to do so in several cases<sub>,</sub> es<sub>p</sub>eciall<sub>y</sub> for activation decrease. In <sub>p</sub>articular<sub>,</sub> our method <sub>y</sub>ields a lar<sub>g</sub>er decrease in activation in all re<sub>p</sub>orted re<sub>g</sub>ions<sub>,</sub> su<sub>pp</sub>ortin<sub>g</sub> the use of an inde<sub>p</sub>endent ne<sub>g</sub>ative <sub>p</sub>rior rather than <sub>s</sub>im<sub>p</sub>l<sub>y</sub> r<sub>eve</sub>r<sub>s</sub>in<sub>g</sub> th<sub>e</sub> m<sub>a</sub>xim<sub>a</sub>l dir<sub>ec</sub>ti<sub>o</sub>n <sub>use</sub>d in th<sub>e</sub> Br<sub>a</sub>inACTIV<sub>.</sub> F<sub>o</sub>r <sub>ac</sub>ti<sub>va</sub>ti<sub>on</sub> i<sub>ncrease,</sub> th<sub>e a</sub>d<sub>van</sub>t<sub>age o</sub>f <sub>our me</sub>th<sub>o</sub>d i<sub>s mos</sub>t <sub>ev</sub>id<sub>en</sub>t in l<sub>ess</sub>-<sub>s</sub>t<sub>u</sub>di<sub>e</sub>d <sub>o</sub>r mix<sub>e</sub>d-<sub>se</sub>l<sub>ec</sub>ti<sub>ve</sub> r<sub>eg</sub>i<sub>o</sub>n<sub>s</sub> <sub>suc</sub>h <sub>as</sub> TE2<sub>p,</sub> IP0<sub>,</sub> <sub>a</sub>nd IP1<sub>,</sub> while <sub>p</sub>erformance is more com<sub>p</sub>arable in stron<sub>g</sub>l<sub>y</sub> cate<sub>g</sub>or<sub>y</sub>- <sub>se</sub>l<sub>ec</sub>ti<sub>ve</sub> r<sub>eg</sub>i<sub>o</sub>n<sub>s suc</sub>h <sub>as</sub> OFA<sub>,</sub> FFA<sub>,</sub> EBA<sub>, a</sub>nd OPA<sub>.</sub> O<sub>u</sub>r m<sub>e</sub>th<sub>o</sub>d also consistentl<sub>y</sub> <sub>p</sub>roduces more realistic visual re<sub>p</sub>resentations<sub>,</sub> as indicated b<sub>y</sub> lower FID scores [22]. Those results demonstrate that, <sub>co</sub>m<sub>pa</sub>r<sub>e</sub>d <sub>w</sub>ith lin<sub>ea</sub>r int<sub>e</sub>r<sub>po</sub>l<sub>a</sub>ti<sub>o</sub>n m<sub>e</sub>th<sub>o</sub>d in Br<sub>a</sub>inACTIV<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h the ne<sub>g</sub>ative direction is sim<sub>p</sub>l<sub>y</sub> the reverse of the <sub>p</sub>ositive one<sub>,</sub> the k<sub>ey</sub> <sub>a</sub>d<sub>van</sub>t<sub>age</sub> <sub>o</sub>f <sub>our</sub> <sub>me</sub>th<sub>o</sub>d i<sub>s</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>ng</sub> <sub>a</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>space</sub> t<sub>o</sub> m<sub>a</sub>k<sub>e</sub> in-di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>sa</sub>m<sub>p</sub>lin<sub>g eas</sub>i<sub>e</sub>r<sub>,</sub> miti<sub>ga</sub>tin<sub>g ou</sub>t-<sub>o</sub>f-di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>pro</sub>bl<sub>em</sub> i<sub>n</sub> li<sub>near</sub> i<sub>n</sub>t<sub>erpo</sub>l<sub>a</sub>ti<sub>on</sub> <sub>me</sub>th<sub>o</sub>d<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>cou</sub>ld <sub>pro</sub>d<sub>uce</sub> <sub>con-</sub> sistentl<sub>y</sub> correct activation chan<sub>g</sub>e<sub>,</sub> and <sub>g</sub>eneratin<sub>g</sub> more realistic i<sub>mages</sub> b<sub>ecause</sub> <sub>samp</sub>l<sub>e</sub> f<sub>rom</sub> <sub>suc</sub>h l<sub>a</sub>t<sub>en</sub>t <sub>space</sub> <sub>w</sub>ill <sub>avo</sub>id <sub>over-s</sub>hift from ori<sub>g</sub>inal ima<sub>g</sub>e re<sub>p</sub>resentations. This is <sub>p</sub>articularl<sub>y</sub> useful when the tar<sub>g</sub>et re<sub>p</sub>resentation is nois<sub>y</sub> or mixed-selective.

(A)  
![](images/fc31c2505e34a86d5943a5faf452abc9de277fa65e05727ec72305422bb99198.jpg)  
Figure 4: Generate preferred images with increasing unit activations in ANNs. (A) The top 10 images demonstrate a notable pattern of “green plant & insect” represented by a unit in the middle layer of the SimCLR model. Our method generates images that better match this pattern and increase activation more than the BrainACTIV [17]. (B) For a mid-layer unit in a more complex Vision Transformer (ViT), our method synthesizes images that can better activate the unit compared with the baseline, although the underlying concept appears more ambiguous.

T<sub>o exam</sub>i<sub>ne</sub> th<sub>e ro</sub>b<sub>us</sub>t<sub>ness o</sub>f th<sub>e pre</sub>di<sub>c</sub>t<sub>or, we</sub> f<sub>ur</sub>th<sub>er compare</sub> activation mo<sup>d</sup>u<sup>l</sup>ation resu<sup>l</sup>ts o<sup>b</sup>taine<sup>d</sup> wit<sup>h</sup> t<sup>h</sup>e origina<sup>l</sup> su<sup>b</sup>jectspeci<sup>fi</sup>c pre<sup>d</sup>ictor an<sup>d</sup> wit<sup>h</sup> a <sup>h</sup>e<sup>ld</sup>-out-su<sup>b</sup>ject pre<sup>d</sup>ictor on t<sup>h</sup>e NSD d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub> Th<sub>e</sub> <sub>resu</sub>lt<sub>s</sub> <sub>are</sub> <sub>s</sub>h<sub>own</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> Th<sub>e</sub> <sub>compara</sub>bl<sub>e</sub> t<sub>ren</sub>d<sub>s</sub> across the two <sub>p</sub>redictors su<sub>gg</sub>est that our conclusions are not <sup>d</sup>riven so<sup>l</sup>e<sup>l</sup>y <sup>b</sup>y a sing<sup>l</sup>e su<sup>b</sup>ject-speci<sup>fi</sup>c mo<sup>d</sup>e<sup>l</sup>.

## 4.3.2 Generate Preferred Representations in Artificial Neural Networks.

Besides biolo<sub>g</sub>ical tar<sub>g</sub>ets<sub>,</sub> we also test our a<sub>pp</sub>roach on <sub>p</sub>urel<sub>y</sub> artificial targets: a mid-level unit inside a SimCLR encoder [8], a <sub>con</sub>t<sub>ras</sub>ti<sub>ve-</sub>l<sub>earn</sub>i<sub>ng</sub> f<sub>ramewor</sub>k <sub>w</sub>ith <sub>a</sub> R<sub>es</sub>N<sub>e</sub>t<sub>-</sub>50 b<sub>ac</sub>kb<sub>one,</sub> <sub>as</sub> <sub>we</sub>ll as a unit from a more com<sub>p</sub>lex Vision Transformer (ViT). Fi<sub>g</sub>ure 4 <sub>s</sub>h<sub>ows</sub> <sub>na</sub>t<sub>ura</sub>l i<sub>mages</sub> th<sub>a</sub>t <sub>pro</sub>d<sub>uce</sub> th<sub>e</sub> l<sub>arges</sub>t <sub>responses</sub> f<sub>rom</sub> th<sub>e</sub> <sub>c</sub>h<sub>ose</sub>n <sub>u</sub>nit<sub>s</sub> in th<sub>ese</sub> ANN<sub>s.</sub> With <sub>ou</sub>r m<sub>o</sub>d<sub>e</sub>l tr<sub>a</sub>in<sub>e</sub>d <sub>o</sub>n <sub>u</sub>nit <sub>ac</sub>ti<sub>va</sub>- tions<sub>,</sub> editin<sub>g</sub> the latent code toward the <sub>p</sub>ositive <sub>p</sub>rior can <sub>g</sub>enerate r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> th<sub>a</sub>t b<sub>e</sub>tt<sub>e</sub>r <sub>ac</sub>ti<sub>va</sub>t<sub>e</sub> th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d ANN <sub>u</sub>nit<sub>s</sub> th<sub>a</sub>n the BrainACTIV [17] in these exam<sub>p</sub>les.

## 4.3.3 Optimize Unit Activations while Preserving Unrelated Features.

T<sub>o</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>e s</sub>t<sub>reng</sub>th <sub>o</sub>f <sub>our s</sub>t<sub>ruc</sub>t<sub>ure</sub>d VAE l<sub>a</sub>t<sub>en</sub>t <sub>space,</sub> <sub>we</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>our mo</sub>d<sub>e</sub>l <sub>w</sub>ith I<sub>ma e</sub>N<sub>e</sub>t d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub> St<sub>an</sub>d<sub>ar</sub>d VAE <sub>w</sub>ith <sub>a s</sub>t<sub>a</sub>nd<sub>a</sub>rd G<sub>auss</sub>i<sub>a</sub>n <sub>p</sub>ri<sub>o</sub>r<sub>.</sub> Th<sub>e</sub>n <sub>we co</sub>n<sub>s</sub>tr<sub>a</sub>in l<sub>a</sub>t<sub>e</sub>nt <sub>o</sub>f VAE <sub>w</sub>ith multiple units activations, and on<sup>l</sup>y optimize t<sup>h</sup>e <sup>l</sup>atent dimensions controlled b<sub>y</sub> a certain <sub>u</sub>nit. Com<sub>p</sub>ared to <sub>v</sub>anilla VAEs<sub>,</sub> o<sub>u</sub>r method achieves more si<sub>g</sub>nificant chan<sub>g</sub>es in unit res<sub>p</sub>onses while <sub>p</sub>reservin<sub>g</sub> unrelated essential ima<sub>g</sub>e characteristics. Besides modi f<sub>y</sub>in<sub>g</sub> latent to <sub>p</sub>rior values<sub>,</sub> which can mani<sub>p</sub>ulate activations and <sub>p</sub>r<sub>ese</sub>r<sub>ve</sub> f<sub>ea</sub>t<sub>u</sub>r<sub>es</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>we</sub> f<sub>u</sub>rth<sub>e</sub>r <sub>op</sub>timiz<sub>e</sub> th<sub>e</sub> VAE’<sub>s</sub> l<sub>a</sub>t<sub>e</sub>nt variables <sub>�</sub> usin<sub>g</sub> <sub>g</sub>radient-based methods<sub>,</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the Adam o<sub>p</sub> timizer to a<sup>d</sup>just � suc<sup>h</sup> t<sup>h</sup>at t<sup>h</sup>e activation � o<sup>f</sup> targete<sup>d</sup> networ<sup>k</sup> units is eit<sup>h</sup>er maximize<sup>d</sup> or minimize<sup>d</sup>. T<sup>h</sup>e optimization o<sup>b</sup>jective combines the desired activation chan<sub>g</sub>e with a KL diver<sub>g</sub>ence (KLD) re<sub>g</sub>ularization term, ensurin<sub>g</sub> that the latent re<sub>p</sub>resentation <sub>rema</sub>i<sub>ns a</sub>li<sub>gne</sub>d <sub>w</sub>ith th<sub>e pr</sub>i<sub>or</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> Th<sub>e resu</sub>lt<sub>s o</sub>f l<sub>a</sub>t<sub>en</sub>t m<sub>o</sub>difi<sub>ca</sub>ti<sub>o</sub>n <sub>a</sub>nd <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>n b<sub>o</sub>th VAE<sub>s a</sub>r<sub>e s</sub>h<sub>ow</sub>n in Fi<sub>gu</sub>r<sub>e</sub> 5<sub>.</sub>

![](images/848e441ed2e0ab91087a4b5a0b9c6af3755d842fb32418e8947f3c9a2a058c81.jpg)  
Figure 5: a) Generating preferred images with diferent methods on ImageNet Dataset. Left 4 columns show optimizing images to increase their activations. Column 1 are four example input images. Column 2-3 are results of our method while column 4 represent optimizing the latent space with a standard VAE. Right 4 columns are results of optimization in the same methods presenting in the same order, with target of decreasing activations. b) shows the shift in the activation value between and the cosine similarity between randomly chosen 100 original images and their optimized results over diferent methods. Compared with baseline VAE, our method can modulate activation better as well as maintain similarity with original images, meaning unrelated features are preserved.

To f<sub>u</sub>rther <sub>v</sub>alidate the efecti<sub>v</sub>eness of o<sub>u</sub>r o<sub>p</sub>timization a<sub>p</sub>- <sub>p</sub>roach. Cosine similarit<sub>y,</sub> com<sub>p</sub>uted usin<sub>g</sub> features extracted from a pre-trained ResNet18 [21] encoder, quanti<sup>fi</sup>es t<sup>h</sup>e preservation o<sup>f</sup> ima<sub>g</sub>e features. Results in Fi<sub>g</sub>ure 5 indicate that our structured landsca<sub>p</sub>e a<sub>pp</sub>roach not onl<sub>y</sub> achieves more robust activation chan<sub>g</sub>es but also maintains similarit<sub>y</sub> to the ori<sub>g</sub>inal ima<sub>g</sub>es com<sub>p</sub>ared to <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rd VAE m<sub>e</sub>th<sub>o</sub>d<sub>s, u</sub>nd<sub>e</sub>r<sub>sco</sub>rin<sub>g</sub> th<sub>e e</sub>fi<sub>cacy o</sub>f in<sub>co</sub>r<sub>po</sub>r<sub>a</sub>tin<sub>g</sub> structured <sub>p</sub>riors for latent o<sub>p</sub>timization.

## 5 Discussion and Limitations

We develo<sub>p</sub>ed Vis-Lens<sub>,</sub> a framework that im<sub>p</sub>oses an activationre<sub>gu</sub>larized <sub>p</sub>rior on a VAE to <sub>y</sub>ield a str<sub>u</sub>ct<sub>u</sub>red latent landsca<sub>p</sub>e and then <sub>p</sub>erforms direct latent editin<sub>g</sub> to steer activations in bi-<sub>o</sub>l<sub>og</sub>i<sub>ca</sub>l <sub>an</sub>d <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l t<sub>arge</sub>t<sub>s.</sub> Thi<sub>s</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>an</sub>d<sub>scape</sub> <sub>can</sub> <sub>a</sub>l<sub>so</sub> <sub>mo</sub>d<sub>u</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>ac</sub>ti<sub>va</sub>ti<sub>on</sub> <sub>o</sub>f t<sub>arge</sub>t <sub>un</sub>it<sub>s</sub> <sub>w</sub>hil<sub>e</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>unre</sub>l<sub>a</sub>t<sub>e</sub>d features. Vis-Lens shows clear advanta<sub>g</sub>es in chan<sub>g</sub>in<sub>g</sub> activations in correct direction<sub>,</sub> kee<sub>p</sub>in<sub>g</sub> ima<sub>g</sub>e realism and decreasin<sub>g</sub> activations <sub>u</sub>nder o<sub>u</sub>r e<sub>v</sub>al<sub>u</sub>ation <sub>p</sub>rotocol. Inhibit acti<sub>v</sub>ation is meanin<sub>g</sub>f<sub>u</sub>l in clinical a<sub>pp</sub>lications<sub>,</sub> reducin<sub>g</sub> abnormall<sub>y</sub> elevated activit<sub>y</sub> in <sub>spec</sub>ifi<sub>c</sub> <sub>c</sub>i<sub>rcu</sub>it<sub>s</sub> h<sub>as</sub> b<sub>een</sub> <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>erapeu</sub>ti<sub>c</sub> b<sub>ene</sub>fit<sub>.</sub> F<sub>or</sub> <sub>examp</sub>l<sub>e,</sub> <sub>pr</sub>i<sub>or</sub> <sub>wor</sub>k h<sub>as</sub> <sub>s</sub>h<sub>own</sub> th<sub>a</sub>t <sub>a</sub>tt<sub>enua</sub>ti<sub>on</sub> <sub>o</sub>f <sub>amyg</sub>d<sub>a</sub>l<sub>a</sub> h<sub>yper-</sub> reactivit<sub>y</sub> is linked to im<sub>p</sub>rovement in anxiet<sub>y</sub> and trauma-related s<sub>y</sub>m<sub>p</sub>toms [29], while reduction of sub<sub>g</sub>enual cin<sub>g</sub>ulate h<sub>yp</sub>eractivit<sub>y</sub> has been associated with antide<sub>p</sub>ressant treatment efects [20]. Thus, althou<sub>g</sub>h our stud<sub>y</sub> is entirel<sub>y</sub> in-silico and does not constitute a medical intervention<sub>,</sub> the abilit<sub>y</sub> to selectivel<sub>y</sub> su<sub>p</sub>- <sub>p</sub>ress tar<sub>g</sub>et-unit res<sub>p</sub>onses ma<sub>y</sub> be relevant for modelin<sub>g</sub> clinicall<sub>y</sub> <sub>mean</sub>i<sub>ng</sub>f<sub>u</sub>l f<sub>orms o</sub>f <sub>neura</sub>l <sub>mo</sub>d<sub>u</sub>l<sub>a</sub>ti<sub>on.</sub>

O<sub>ur</sub> f<sub>ramewor</sub>k <sub>can a</sub>l<sub>so ex</sub>t<sub>en</sub>d t<sub>o ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>s as</sub> a controlled settin<sub>g</sub> for stud<sub>y</sub>in<sub>g</sub> feature visualization. Com<sub>p</sub>ared <sub>w</sub>ith th<sub>e</sub> li<sub>near me</sub>th<sub>o</sub>d<sub>,</sub> Vi<sub>s-</sub>L<sub>ens</sub> d<sub>oes no</sub>t <sub>assume</sub> th<sub>a</sub>t <sub>ac</sub>ti<sub>va</sub>ti<sub>on</sub> increase and decrease follo<sub>w</sub> a sin<sub>g</sub>le linear direction in CLIP s<sub>p</sub>ace. Instead<sub>,</sub> it learns activation-s<sub>p</sub>ecific re<sub>g</sub>ions in a structured latent <sub>space, an</sub>d <sub>ena</sub>bl<sub>es more con</sub>t<sub>ro</sub>ll<sub>a</sub>bl<sub>e samp</sub>li<sub>ng, w</sub>hi<sub>c</sub>h i<sub>s espec</sub>i<sub>a</sub>ll<sub>y</sub> <sub>use</sub>f<sub>u</sub>l <sub>w</sub>h<sub>en</sub> <sub>neura</sub>l <sub>responses</sub> <sub>are</sub> <sub>no</sub>i<sub>sy</sub> <sub>an</sub>d <sub>m</sub>i<sub>xe</sub>d<sub>-se</sub>l<sub>ec</sub>ti<sub>ve.</sub>

There are also limitations to our work. The ima<sub>g</sub>e <sub>g</sub>enerator and the CLIP encoder are frozen<sub>,</sub> so biases inherited from their <sub>p</sub>re trainin<sub>g</sub> data ma<sub>y</sub> <sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ate into the <sub>g</sub>enerated <sub>p</sub>referred ima<sub>g</sub>es.

I<sub>n</sub> th<sub>e</sub> f<sub>u</sub>t<sub>ure,</sub> <sub>we</sub> <sub>may</sub> <sub>sca</sub>l<sub>e</sub> th<sub>e</sub> f<sub>ramewor</sub>k t<sub>o</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>mo</sub>d<sub>a</sub>l<sub>-</sub> iti<sub>es suc</sub>h <sub>as</sub> EEG<sub>,</sub> MEG <sub>o</sub>r l<sub>a</sub>r<sub>ge</sub>r-<sub>sca</sub>l<sub>e</sub> br<sub>a</sub>in d<sub>a</sub>t<sub>ase</sub>t<sub>s, a</sub>nd m<sub>ove</sub> be<sub>y</sub>ond in-silico evaluation b<sub>y</sub> testin<sub>g</sub> the <sub>g</sub>enerated <sub>p</sub>referred im-<sub>ages</sub> i<sub>n</sub> h<sub>uman</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s;</sub> <sub>suc</sub>h <sub>c</sub>l<sub>ose</sub>d<sub>-</sub>l<sub>oop</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>wou</sub>ld f<sub>ur</sub>th<sub>er con</sub>fi<sub>rm</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>veness o</sub>f <sub>our</sub> f<sub>ramewor</sub>k <sub>an</sub>d b<sub>roa</sub>d<sub>en</sub> it<sub>s</sub> <sub>p</sub>otential use in brain-com<sub>p</sub>uter-interface a<sub>pp</sub>lications.

## References

[1] I<sub>g</sub>or Abramov. 2025. EEG-Driven Ima<sub>g</sub>e Reconstruction with Salienc<sub>y</sub>-Guided Difusion Models. In Proceedings ofthe 33rd ACM International Conference on Multimedia (MM ’25). ACM 13522–13524. doi:10.1145/3746027.3754476

[2] Emil<sub>y</sub> J. Allen, Ghislain St-Yves, Yihan Wu, Jesse L. Breedlove, Lo<sub>g</sub>an T. Dowdle, Brad Caron, Franco Pestilli, Ian Charest, J. Benjamin Hutchin-<sub>so</sub>n<sub>,</sub> Th<sub>o</sub>m<sub>as</sub> N<sub>ase</sub>l<sub>a</sub>ri<sub>s, a</sub>nd K<sub>e</sub>ndri<sub>c</sub>k K<sub>ay.</sub> 2021<sub>.</sub> A m<sub>ass</sub>i<sub>ve</sub> 7T fMRI dataset to bridge cognitive and computational neuroscience. bioRxiv (2021). arXiv:htt<sub>p</sub>s://www.biorxiv.or<sub>g</sub>/content/earl<sub>y</sub>/2021/02/22/2021.02.22.432340.full.<sub>p</sub>d doi:10.1101/2021.02.22.432340

[3] Martin Arjovsk<sub>y</sub>, Soumith Chintala, and Léon Bottou. 2017. Wasserstein GAN. arXiv preprint arXiv:1701.07875 (2017)

[4] Pouya Bashivan, Kohitij Kar, and James J DiCarlo. 2019. Neural population control via dee ima e s nthesis. Science 364, 6439 (2019), eaav9436.

[5] David Bau et al. 2017. Network Dissection: Quantif<sub>y</sub>in<sub>g</sub> Inter<sub>p</sub>retabilit<sub>y</sub> of Dee<sub>p</sub> Visual Representations. In Conference on Computer Vision and Pattern Recognition (CVPR).

[6] Vinamra Benara, Chandan Sin<sub>g</sub>h, John X. Morris, Richard Antonello, Ion Stoica, Alexander G. Huth, and Jianfen<sub>g</sub> Gao. 2024. Craftin<sub>g</sub> Inter<sub>p</sub>retable Embeddin<sub>g</sub>s b<sub>y</sub> Askin<sub>g</sub> LLMs Questions. arXiv:2405.16714 [cs.CL] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2405. 16714

[7] Yohann Benchetrit, Hubert Banville, and Jean-Rémi Kin<sub>g</sub>. 2024. Brain decodin<sub>g</sub>: toward real-time reconstruction of visual perception. In International Conference on Learning Representations (ICLR).

[8] Tin<sub>g</sub> Chen, Simon Kornblith, Mohammad Norouzi, and Geofre<sub>y</sub> Hinton. 2020. A Sim<sub>p</sub>l<sub>e</sub> Fr<sub>a</sub>m<sub>ewo</sub>rk f<sub>o</sub>r C<sub>o</sub>ntr<sub>as</sub>ti<sub>ve</sub> L<sub>ea</sub>rnin<sub>g o</sub>f Vi<sub>sua</sub>l R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub> In International Conference on Machine Learning (ICML).

[9] Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. 2018. Pixel-SNAIL: An Improved Autoregressive Generative Model. In Proceedings of the International Conference on Machine Learning (ICML).

[10] Zijiao Chen, Jiaxin Qing, Tiange Xiang, Wan Lin Yue, and Juan Helen Zhou. 2023. Seein<sub>g</sub> Be<sub>y</sub>ond the Brain: Conditional Dif<sub>u</sub>sion Model <sub>w</sub>ith S<sub>p</sub>arse Masked Modelin<sub>g</sub> for Vision Decodin<sub>g</sub>. arXiv:2211.06956 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/ 2211.06956

[11] T Dado, P Pa<sub>p</sub>ale, et al. 2024. Brain2GAN: Feature-Disentan<sub>g</sub>led Neural Encodin<sub>g</sub> and Decoding of Visual Perception in the Primate Brain. PLoS Computational Biology (2024). https://journals.plos.org/ploscompbiol/article?id=10.1371/journal. <sub>p</sub>cbi.1012058

[12] Carlos de la Torre-Ortiz and Tuukka Ruotsalo. 2024. Perce<sub>p</sub>tual Visual Similarit<sub>y</sub> fr<sub>o</sub>m EEG: Pr<sub>e</sub>di<sub>c</sub>ti<sub>o</sub>n <sub>a</sub>nd Im<sub>age</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> 11146-11155 <sub>pages.</sub> d<sub>o</sub>i:10<sub>.</sub>1145/ 3664647.3685508

[13] Jia Den<sub>g</sub>, Wei Don<sub>g</sub>, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Ima geNet: A large-scale hierarchical image database. In 2009IEEE Conference on Computer Vision and Pattern Recognition. 248–255. doi:10.1109/CVPR.2009.5206848

[14] Zhiwei Din<sub>g</sub>, Dat Tran, Ka<sub>y</sub>la Ponder, Erick Cobos, Zhuokun Din<sub>g</sub>, Paul Fahe<sub>y</sub>, Eric Wan<sub>g</sub>, Taliah Muhammad, Jiakun Fu, Santia<sub>g</sub>o Cadena, Stelios Pa<sub>p</sub>ado<sub>p</sub>oulos, Saumil Patel, Katrin Franke, Jacob Reimer, Fabian Sinz, Alexander Ecker, Xa<sub>q</sub> Pitk<sub>ow, a</sub>nd Andr<sub>eas</sub> T<sub>o</sub>li<sub>as.</sub> 2023<sub>.</sub> Bi<sub>pa</sub>rtit<sub>e</sub> in<sub>va</sub>ri<sub>a</sub>n<sub>ce</sub> in m<sub>ouse p</sub>rim<sub>a</sub>r<sub>y v</sub>i<sub>sua</sub>l cortex. doi:10.1101/2023.03.15.532836

[15] Dumitru Erhan, Yoshua Ben<sub>g</sub>io, Aaron Courville, and Pascal Vincent. 2009. Visualizing higher-layer features of a deep network. University ofMontreal (2009).

[16] Matteo Ferrante, Tommaso Boccato, Luca Passamonti, and Nicola Toschi. 2024. Retrievin<sub>g</sub> and reconstructin<sub>g</sub> conce<sub>p</sub>tuall<sub>y</sub> similar ima<sub>g</sub>es from fMRI with latent difusion models and a neuro-inspired brain decoding model. Journal ofNeural Engineering 21, 4 (2024), 046001.

[17] Die<sub>g</sub>o Garcia Cerdas, Christina Sartzetaki, Ma<sub>g</sub>nus Petersen, Gemma Roi<sub>g</sub>, Pas <sub>ca</sub>l M<sub>e</sub>tt<sub>es,</sub> <sub>a</sub>nd Iri<sub>s</sub> Gr<sub>oe</sub>n<sub>.</sub> 2025<sub>.</sub> Br<sub>a</sub>inACTIV: Id<sub>e</sub>ntif<sub>y</sub>in<sub>g</sub> <sub>v</sub>i<sub>suo</sub>-<sub>se</sub>m<sub>a</sub>nti<sub>c</sub> <sub>p</sub>ro<sub>p</sub>erties drivin<sub>g</sub> cortical selectivit<sub>y</sub> usin<sub>g</sub> difusion-based ima<sub>g</sub>e mani<sub>p</sub>ula tion. In Proceedings of the International Conference on Learning Representations. htt<sub>p</sub>s://o<sub>p</sub>enreview.net/forum?id=CGON8Btleu

[18] Ian Goodfellow, Jean Pou<sub>g</sub>et-Abadie, Mehdi Mirza, Bin<sub>g</sub> Xu, David Warde-Farle<sub>y</sub>, S<sup>h</sup>erji<sup>l</sup> Ozair, Aaron Courvi<sup>ll</sup>e, an<sup>d</sup> Yos<sup>h</sup>ua Bengio. 2014. Generative a<sup>d</sup>versaria<sup>l</sup> nets. Advances in Neural Information Processing Systems (NeurIPS) (2014)

[19] Zijin Gu, Keith Wakefield Jamison, Meenakshi Khosla, Emily J. Allen, Yihan Wu, Ghi<sub>s</sub>l<sub>a</sub>in St-Y<sub>ves,</sub> Th<sub>o</sub>m<sub>as</sub> N<sub>ase</sub>l<sub>a</sub>ri<sub>s,</sub> K<sub>e</sub>ndri<sub>c</sub>k K<sub>ay,</sub> M<sub>e</sub>rt R<sub>.</sub> S<sub>a</sub>b<sub>u</sub>n<sub>cu,</sub> <sub>a</sub>nd Am<sub>y</sub> K<sub>uceyes</sub>ki<sub>.</sub> 2022<sub>.</sub> N<sub>eu</sub>r<sub>o</sub>G<sub>e</sub>n: A<sub>c</sub>ti<sub>va</sub>ti<sub>o</sub>n <sub>op</sub>timiz<sub>e</sub>d im<sub>age sy</sub>nth<sub>es</sub>i<sub>s</sub> f<sub>o</sub>r di<sub>scove</sub>r<sub>y</sub> neuroscience. NeuroImage 247 (2022), 118812. doi:10.1016/j.neuroimage.2021. 118812

[20] Ita<sub>y</sub> Hadas, Yinmin<sub>g</sub> Sun, Pantelis Lioumis, Reza Zomorrodi, Brett Jones, Da<sub>p</sub>hne Voineskos, Jonathan Downar, Paul Fitz<sub>g</sub>erald, Daniel Blumber<sub>g</sub>er, and Zafiris D<sub>as</sub>k<sub>a</sub>l<sub>a</sub>ki<sub>s.</sub> 2019<sub>.</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f R<sub>epe</sub>titi<sub>ve</sub> Tr<sub>a</sub>n<sub>sc</sub>r<sub>a</sub>ni<sub>a</sub>l M<sub>ag</sub>n<sub>e</sub>ti<sub>c</sub> Stim<sub>u</sub>l<sub>a</sub>ti<sub>o</sub>n Treatment Wit<sup>h</sup> Su<sup>b</sup>genua<sup>l</sup> Cingu<sup>l</sup>ate Hyperactivity in Patients Wit<sup>h</sup> Major Depressive Disorder: A Secondary Analysis ofa Randomized Clinical Trial. JAMA Network Open 2 (06 2019), e195578. doi:10.1001/jamanetworkopen.2019.5578

[21] Kaimin<sub>g</sub> He, Xian<sub>gy</sub>u Zhan<sub>g</sub>, Shao<sub>q</sub>in<sub>g</sub> Ren, and Jian Sun. 2016. Dee<sub>p</sub> Resid ual Learning for Image Recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 770–778.

[22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and S<sub>epp</sub> H<sub>oc</sub>hr<sub>e</sub>it<sub>e</sub>r<sub>.</sub> 2017<sub>.</sub> GAN<sub>s</sub> Tr<sub>a</sub>in<sub>e</sub>d b<sub>y a</sub> T<sub>wo</sub> Tim<sub>e</sub>-S<sub>ca</sub>l<sub>e</sub> U<sub>p</sub>d<sub>a</sub>t<sub>e</sub> R<sub>u</sub>l<sub>e</sub> C<sub>o</sub>n<sub>ve</sub>r<sub>ge</sub> to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems, I. G<sub>uy</sub>on<sub>,</sub> U. Von L<sub>u</sub>xb<sub>u</sub>r<sub>g,</sub> S. Ben<sub>g</sub>io<sub>,</sub> H. Wallach<sub>,</sub> R. Fer<sub>gu</sub>s<sub>,</sub> S. Vish<sub>w</sub>anathan<sub>,</sub> and R. Garnett (Eds.), Vol. 30. Curran Associates, Inc. htt<sub>p</sub>s://<sub>p</sub>roceedin<sub>g</sub>s.neuri<sub>p</sub>s. <sub>cc</sub>/<sub>paper\_</sub>fil<sub>es</sub>/<sub>paper</sub>/2017/fil<sub>e</sub>/8<sub>a</sub>1d694707<sub>e</sub>b0f<sub>e</sub>f<sub>e</sub>65871369074926d<sub>-</sub>P<sub>aper.p</sub>df

[23] Jonathan Ho, AjayJain, and Pieter Abbeel. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems (NeurIPS).

[24] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, <sub>an</sub>d Ti<sub>m</sub> S<sub>a</sub>li<sub>mans.</sub> 2022<sub>.</sub> C<sub>asca</sub>d<sub>e</sub>d dif<sub>us</sub>i<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>or</sub> hi<sub>g</sub>h fid<sub>e</sub>lit<sub>y</sub> i<sub>mage</sub> generation. Journal ofMachine Learning Research 23, 47 (2022), 1–33.

[25] Tomo<sub>y</sub>asu Horikawa and Yuki<sub>y</sub>asu Kamitani. 2017. Generic decodin<sub>g</sub> of seen and imagined objects using hierarchical visual features. Nature Communications 8 (2017), 15037.

[26] Shuo Huan<sub>g</sub>, Wei Shao, Mei-Lin<sub>g</sub> Wan<sub>g</sub>, and Dao-Qian<sub>g</sub> Zhan<sub>g</sub>. 2021. fMRI-based D<sub>eco</sub>din<sub>g o</sub>f Vi<sub>sua</sub>l Inf<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n fr<sub>o</sub>m H<sub>u</sub>m<sub>a</sub>n Br<sub>a</sub>in A<sub>c</sub>ti<sub>v</sub>it<sub>y</sub>: A Bri<sub>e</sub>f R<sub>ev</sub>i<sub>ew.</sub> International Journal ofAutomation and Computing 18 (01 2021). doi:10.1007/ s11633-020-1263-

[27] David H Hubel and Torsten N Wiesel. 1962. Rece<sub>p</sub>tive fields, binocular interaction and functional architecture in the cat’s visual cortex. The Journal of physiology 160, 1 (1962), 106.

[28] Jin<sub>gy</sub>an<sub>g</sub> Huo, Yikai Wan<sub>g</sub>, Yanwei Fu, and Jianfen<sub>g</sub> Fen<sub>g</sub>. 2026. The Pictorial Cortex: Zero-S<sup>h</sup>ot Cross-Su<sup>b</sup>ject <sup>f</sup>MRI-to-Ima e Reconstruction via Com ositiona Latent Modelin . arXiv:2601.15071 [cs.CV] htt s://arxiv.or /abs/2601.15071

[29] Maria Ironside, Michael Brownin<sub>g</sub>, Tahereh Ansari, Christo<sub>p</sub>her Harve<sub>y</sub>, Mama Sekyi-Djan, Sonia Bishop, Catherine Harmer, and Jacinta O’Shea. 2019. Efect of Prefrontal Cortex Stim<sub>u</sub>lation on Re<sub>gu</sub>lation of Am<sub>yg</sub>dala Res<sub>p</sub>onse to Threat in Individuals With Trait Anxiety: A Randomized Clinical Trial. JAMA Psychiatry 76 (01 2019), 71–78. doi:10.1001/jama s chiatr .2018.2172

[30] Diederik P Kin<sub>g</sub>ma and Max Wellin<sub>g</sub>. 2013. Auto-encodin<sub>g</sub> variational ba<sub>y</sub>es. arXiv preprint arXiv:1312.6114 (2013).

[31] Paulo H. Leocadio. 2025. Hugging Face Difusers - Chapter 07. Hugging Face (Jan. 2025). doi:10.22541/au.173627634.49343603/v1

[32] Dongyang Li, Haoyang Qin, Mingyang Wu, Chen Wei, and Quanying Liu. 2025. Br<sub>a</sub>inFLORA: Un<sub>cove</sub>rin<sub>g</sub> Br<sub>a</sub>in C<sub>o</sub>n<sub>cep</sub>t R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n <sub>v</sub>i<sub>a</sub> M<sub>u</sub>ltim<sub>o</sub>d<sub>a</sub>l N<sub>eu</sub>r<sub>a</sub>l Embeddin<sub>g</sub>s. arXiv:2507.09747 [cs.NE] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2507.09747

[33] Dongyang Li, Chen Wei, Shiying Li, Jiachen Zou, Haoyang Qin, and Quanying Li<sub>u.</sub> 2024<sub>.</sub> Vi<sub>sua</sub>l D<sub>eco</sub>din<sub>g a</sub>nd R<sub>eco</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n <sub>v</sub>i<sub>a</sub> EEG Emb<sub>e</sub>ddin<sub>gs w</sub>ith G<sub>u</sub>id<sub>e</sub>d Difusion. arXiv:2403.07721 [cs.HC] htt s://arxiv.or /abs/2403.07721

[34] Tsun<sub>g</sub>-Yi Lin, Michael Maire, Ser<sub>g</sub>e Belon<sub>g</sub>ie, Lubomir Bourdev, Ross Girshick, James Ha<sub>y</sub>s, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. 2015. Microsoft COCO: Common Objects in Context. arXiv:1405.0312 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/1405.0312

[35] Yulon<sub>g</sub> Liu, Yon<sub>gq</sub>ian<sub>g</sub> Ma, Wei Zhou, Guibo Zhu, and Nannin<sub>g</sub> Zhen<sub>g</sub>. 2023. Braincli<sub>p</sub>: Brid<sub>g</sub>in<sub>g</sub> brain and <sub>v</sub>is<sub>u</sub>al-lin<sub>gu</sub>istic re<sub>p</sub>resentation <sub>v</sub>ia cli<sub>p</sub> for <sub>g</sub>eneric natural visual stimulus decoding. arXiv preprint arXiv:2302.12971 (2023).

[36] Andrew F. Luo, Mar<sub>g</sub>aret M. Henderson, Michael J. Tarr, and Leila Wehbe. 2024. BrainSCUBA: Fine-Grained Nat<sub>u</sub>ral Lan<sub>gu</sub>a<sub>g</sub>e Ca<sub>p</sub>tions of Vis<sub>u</sub>al Cortex Selec tivit<sub>y</sub>. arXiv:2310.04420 [cs.LG] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2310.04420

[37] Andrew F. Luo, Mar<sub>g</sub>aret M. Henderson, Leila Wehbe, and Michael J. Tarr. 2023. Br<sub>a</sub>in Dif<sub>us</sub>i<sub>o</sub>n f<sub>o</sub>r Vi<sub>sua</sub>l Ex<sub>p</sub>l<sub>o</sub>r<sub>a</sub>ti<sub>o</sub>n: C<sub>o</sub>rti<sub>ca</sub>l Di<sub>scove</sub>r<sub>y us</sub>in<sub>g</sub> L<sub>a</sub>r<sub>ge</sub> S<sub>ca</sub>l<sub>e</sub> Generative Models. arXiv:2306.03089 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2306.03089

[38] Jun Ma and Tuukka Ruotsalo. 2024. Co<sub>g</sub>nition-Su<sub>p</sub>ervised Salienc<sub>y</sub> Detection: Contrasting EEG Signals and Visual Stimuli. In Proceedings ofthe 32nd ACM International Conference on Multimedia (Melbourne VIC, Australia) (MM ’24). A<sub>ssoc</sub>i<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r C<sub>o</sub>m<sub>pu</sub>tin<sub>g</sub> M<sub>ac</sub>hin<sub>e</sub>r<sub>y,</sub> N<sub>ew</sub> Y<sub>o</sub>rk<sub>,</sub> NY<sub>,</sub> USA<sub>,</sub> 7744–7753<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub> 1145/3664647.3681037

[39] Aravindh Mahendran and Andrea Vedaldi. 2015. Understandin<sub>g</sub> dee<sub>p</sub> ima<sub>g</sub>e re<sub>p</sub>- resentations by inverting them. In Proceedings ofthe IEEE conference on computer vision and pattern recognition (CVPR). 5188–5196.

[40] Weijian Mai, Jiamin Wu, Yu Zhu, Zhouhen<sub>g</sub> Yao, Don<sub>g</sub>zhan Zhou, Andrew F. Luo, Qihao Zheng, Wanli Ouyang, and Chunfeng Song. 2025. SynBrain: En h<sub>a</sub>n<sub>c</sub>in<sub>g</sub> Vi<sub>sua</sub>l-t<sub>o</sub>-fMRI S<sub>y</sub>nth<sub>es</sub>i<sub>s</sub> <sub>v</sub>i<sub>a</sub> Pr<sub>o</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n L<sub>ea</sub>rnin<sub>g.</sub> arXiv:2508.10298 [cs.LG] htt s://arxiv.or /abs/2508.10298

[41] David Marr. 2010. Vision: A computational investigation into the human representation and processing of visual information. MIT press.

[42] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, <sub>a</sub>nd St<sub>e</sub>f<sub>a</sub>n<sub>o</sub> Erm<sub>o</sub>n<sub>.</sub> 2022<sub>.</sub> SDEdit: G<sub>u</sub>id<sub>e</sub>d Im<sub>age</sub> S<sub>y</sub>nth<sub>es</sub>i<sub>s a</sub>nd Editin<sub>g w</sub>ith Stochastic Diferential E<sub>q</sub>uations. arXiv:2108.01073 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/ abs/2108.01073

[43] Mehdi Mirza and Simon Osindero. 2014. Conditional Generative Adversarial Nets. In arXiv preprint arXiv:1411.1784.

[44] Takeru Mi<sub>y</sub>ato, Toshiki Kataoka, Masanori Ko<sub>y</sub>ama, and Yuichi Yoshida. 2018. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations (ICLR). htt s://arxiv.or /abs/1802.05957

[45] Alexander Mordvintsev, Chris Olah, and Mike T<sub>y</sub>ka. 2015. Ince<sub>p</sub>tionism: Goin<sub>g</sub> deeper into neural networks. Google Research Blog 5 (2015).

[46] Alexander Mordvintsev, Chris Olah, and Mike T<sub>y</sub>ka. 2016. Dee<sub>p</sub>Dream—A code example for visualizing Neural Networks. Google Research Blog (2016).

[47] N. Murt<sub>y</sub>, Pou<sub>y</sub>a Bashivan, Alex Abate, James Dicarlo, and Nanc<sub>y</sub> Kanwisher. 2021. Com<sub>p</sub>utational models of cate<sub>g</sub>or<sub>y</sub>-selective brain re<sub>g</sub>ions enable hi<sub>g</sub>hthroughput tests of selectivity. Nature Communications 12 (09 2021), 5540. doi:10. 1038/s41467-021-25409-6

[48] Thomas Naselaris, Kendrick N Ka<sub>y</sub>, Shinji Nishimoto, and Jack L Gallant. 2011. Encoding and decoding in fMRI. Neuroimage 56, 2 (2011), 400–410.

[49] Anh N<sub>g</sub>u<sub>y</sub>en, Jef Clune, Yoshua Ben<sub>g</sub>io, Alexe<sub>y</sub> Dosovitski<sub>y</sub>, and Jason Yosinski. 2017<sub>.</sub> Pl<sub>ug</sub> & Pl<sub>ay</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>ve</sub> N<sub>e</sub>t<sub>wo</sub>rk<sub>s</sub>: C<sub>o</sub>nditi<sub>o</sub>n<sub>a</sub>l It<sub>e</sub>r<sub>a</sub>ti<sub>ve</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f Images in Latent Space. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 4467–4477.

[50] Anh N<sub>g</sub>u<sub>y</sub>en, Jason Yosinski, and Jef Clune. 2015. Dee<sub>p</sub> neural networks are easily fooled: High confidence predictions for unrecognizable images. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 427–436.

[51] Anh N<sub>g</sub>u<sub>y</sub>en, Jason Yosinski, and Jef Clune. 2016. S<sub>y</sub>nthesizin<sub>g</sub> the <sub>p</sub>referred inputs for neurons in neural networks via deep generator networks. In Advances in Neural Information Processing Systems (NeurIPS). 3387–3395.

[52] Alex Nichol, Prafulla Dhariwal, Adit<sub>y</sub>a Ramesh, Pranav Sh<sub>y</sub>am, Pamela Mishkin, B<sub>o</sub>b M<sub>c</sub>Gr<sub>ew,</sub> Il <sub>a</sub> S<sub>u</sub>t<sub>s</sub>k<sub>eve</sub>r<sub>, a</sub>nd M<sub>a</sub>rk Ch<sub>e</sub>n<sub>.</sub> 2022<sub>.</sub> GLIDE: T<sub>owa</sub>rd<sub>s</sub> Ph<sub>o</sub>- t<sub>o</sub>r<sub>ea</sub>li<sub>s</sub>ti<sub>c</sub> Im<sub>age</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd Editin<sub>g w</sub>ith T<sub>e</sub>xt-G<sub>u</sub>id<sub>e</sub>d Dif<sub>us</sub>i<sub>o</sub>n M<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> arXiv:2112.10741 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2112.10741

[53] Shinji Nishimoto, An T Vu, Thomas Naselaris, Yuval Benjamini, Bin Yu, and Jack L Gallant. 2011. Reconstructin visual ex eriences from brain activit evoked by natural movies. Current Biology 21, 19 (2011), 1641–1646.

[54] Tormod Ø <sub>yg</sub>ard. 2015. Gradient blurrin<sub>g</sub> for im<sub>p</sub>roved feature visualization. Personal communication (2015).

[55] Chris Olah et al. 2017. Feature Visualization. Distill (2017).

[56] Bruno A Olshausen and David J Field. 2004. S<sub>p</sub>arse codin<sub>g</sub> of sensor<sub>y</sub> in<sub>p</sub>uts. Current opinion in neurobiology 14, 4 (2004), 481–487.

[57] Maxime O<sub>q</sub>uab, Timothée Darcet, Théo Moutakanni, Hu<sub>y</sub> Vo, Marc Szafraniec, V<sub>as</sub>il Kh<sub>a</sub>lid<sub>ov,</sub> Pi<sub>e</sub>rr<sub>e</sub> F<sub>e</sub>rn<sub>a</sub>nd<sub>e</sub>z<sub>,</sub> D<sub>a</sub>ni<sub>e</sub>l H<sub>a</sub>ziz<sub>a,</sub> Fr<sub>a</sub>n<sub>c</sub>i<sub>sco</sub> M<sub>assa,</sub> Al<sub>aae</sub>ldin E<sup>l</sup>-Nou<sup>b</sup>y, Ma<sup>h</sup>mou<sup>d</sup> Assran, Nico<sup>l</sup>as Ba<sup>ll</sup>as, Wojciec<sup>h</sup> Ga<sup>l</sup>u<sup>b</sup>a, Russe<sup>ll</sup> Howes, P<sub>o</sub>-Y<sub>ao</sub> H<sub>ua</sub>n<sub>g,</sub> Sh<sub>a</sub>n<sub>g</sub>-W<sub>e</sub>n Li<sub>,</sub> I<sub>s</sub>h<sub>a</sub>n Mi<sub>s</sub>r<sub>a,</sub> Mi<sub>c</sub>h<sub>ae</sub>l R<sub>a</sub>bb<sub>a</sub>t<sub>,</sub> V<sub>asu</sub> Sh<sub>a</sub>rm<sub>a,</sub> G<sub>a</sub>bri<sub>e</sub>l S nnaeve, Hu Xu, Hervé Je ou, Julien Mairal, Patrick Labatut, Armand Joulin, an<sup>d</sup> Piotr Bojanows<sup>k</sup>i. 2024. DINOv2: Learning Ro<sup>b</sup>ust Visua<sup>l</sup> Features wit<sup>h</sup>out Su<sub>p</sub>ervision. arXiv:2304.07193 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2304.07193

[58] Furkan Ozcelik and Rufin VanRullen. 2023. Natural scene reconstruction from fMRI si<sub>g</sub>nals usin<sub>g g</sub>enerative latent difusion. arXiv:2303.05334 [cs.CV] htt<sub>p</sub>s: //arxiv.or<sub>g</sub>/abs/2303.05334

[59] Paolo Pa<sub>p</sub>ale, Daniela De Luca, and Pieter R. Roelfsema. 2024. Dee<sub>p g</sub>enerative networks reveal the tunin<sub>g</sub> of neurons in IT and <sub>p</sub>redict their influence on visual <sub>p</sub>erce<sub>p</sub>tion. bioRxiv (2024). arXiv:htt<sub>p</sub>s://www.biorxiv.or<sub>g</sub>/content/earl<sub>y</sub>/2024/10/12/2024.10.09.617382.full.<sub>p</sub>d doi:10.1101/2024.10.09.617382

[60] Paweł A. Pierzchlewicz, Konstantin Willeke, Arne Nix, Pavithra Elumalai, Kelli R<sub>es</sub>ti<sub>vo,</sub> T<sub>o</sub>ri Shinn<sub>,</sub> C<sub>a</sub>t<sub>e</sub> N<sub>ea</sub>ll<sub>ey,</sub> G<sub>a</sub>bri<sub>e</sub>ll<sub>e</sub> R<sub>o</sub>dri<sub>gue</sub>z<sub>,</sub> S<sub>au</sub>mil P<sub>a</sub>t<sub>e</sub>l<sub>,</sub> K<sub>a</sub>trin Fr<sub>a</sub>nk<sub>e,</sub> Andr<sub>eas</sub> T<sub>o</sub>li<sub>as, a</sub>nd F<sub>a</sub>bi<sub>a</sub>n Sinz<sub>.</sub> 2023<sub>.</sub> En<sub>e</sub>r<sub>gy</sub> G<sub>u</sub>id<sub>e</sub>d Dif<sub>us</sub>i<sub>o</sub>n f<sub>o</sub>r Generatin<sub>g</sub> Ne<sub>u</sub>rall<sub>y</sub> Excitin<sub>g</sub> Ima<sub>g</sub>es. doi:10.1101/2023.05.18.541176

[61] D Qian, H Zen<sub>g</sub>, et al. 2024. NeuroDM: Decodin<sub>g</sub> and Visualizin<sub>g</sub> Human Brain Activity with EEG-Guided Difusion Model. Computer Methods and Programs in Biomedicine (2024). https://www.sciencedirect.com/science/article/pii/ S0169260724002098

[62] Weikang Qiu, Zheng Huang, Haoyu Hu, Aosong Feng, Yujun Yan, and Rex Ying. 2025. Min<sup>d</sup>LLM: A Su<sup>b</sup>ject-Agnostic an<sup>d</sup> Versati<sup>l</sup>e Mo<sup>d</sup>e<sup>l f</sup>or <sup>f</sup>MRI-to-Text Decodin<sub>g</sub>. arXiv:2502.15786 [<sub>q</sub>-bio.NC] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2502.15786

[63] Rodrigo Quian Quiroga, Leila Reddy, Gabriel Kreiman, Christof Koch, and Itzhak Fri<sub>e</sub>d<sub>.</sub> 2005<sub>.</sub> In<sub>va</sub>ri<sub>a</sub>nt <sub>v</sub>i<sub>sua</sub>l r<sub>e</sub> r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n b <sub>s</sub>in l<sub>e</sub> n<sub>eu</sub>r<sub>o</sub>n<sub>s</sub> in th<sub>e</sub> h<sub>u</sub>m<sub>a</sub>n brain. Nature 435, 7045 (2005), 1102–1107.

[64] Alec Radford, Jon<sub>g</sub> Wook Kim, Chris Hallac<sub>y</sub>, Adit<sub>y</sub>a Ramesh, Gabriel Goh, Sandhini A<sub>g</sub>arwal, Girish Sastr<sub>y</sub>, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krue<sub>g</sub>er<sub>,</sub> and Il<sub>y</sub>a Sutskever. 2021. Learnin<sub>g</sub> Transferable Visual Models From Natural Language Supervision. In Proceedings of the International Conference on Machine Learning (ICML). https://arxiv.org/abs/2103.00020

[65] Mattia Ri<sub>g</sub>otti, O. Barak, M. Warden, Xiao-Jin<sub>g</sub> Wan<sub>g</sub>, N. Daw, E. Miller, and Stefano F<sub>u</sub>si. 2013. The im<sub>p</sub>ortance of mixed selecti<sub>v</sub>it<sub>y</sub> in com<sub>p</sub>lex co<sub>g</sub>niti<sub>v</sub>e tasks. Nature 497 (2013), 585–590. doi:10.1038/nature12160

[66] Amanda Robinson, Genevieve Quek, and Thomas Carlson. 2023. Visual Re<sub>p</sub>resentations: Insights from Neural Decoding. Annual Review ofVision Science 9 (03 2023). doi:10.1146/annurev-vision-100120-025301

[67] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Omm<sub>e</sub>r<sub>.</sub> 2022<sub>.</sub> Hi<sub>g</sub>h-R<sub>eso</sub>l<sub>u</sub>ti<sub>o</sub>n Im<sub>age</sub> S<sub>y</sub>nth<sub>es</sub>i<sub>s w</sub>ith L<sub>a</sub>t<sub>e</sub>nt Dif<sub>us</sub>i<sub>o</sub>n M<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> arXiv:2112.10752 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2112.10752

[68] Christo<sub>p</sub>h Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wi<sub>g</sub>htm<sub>a</sub>n<sub>,</sub> M<sub>e</sub>hdi Ch<sub>e</sub>rti<sub>,</sub> Th<sub>eo</sub> C<sub>oo</sub>mb<sub>es,</sub> A<sub>a</sub>r<sub>us</sub>h K<sub>a</sub>tt<sub>a,</sub> Cl<sub>ay</sub>t<sub>o</sub>n M<sub>u</sub>lli<sub>s,</sub> Mit<sub>c</sub>h<sub>e</sub>ll W<sub>o</sub>rt<sub>s</sub>m<sub>a</sub>n<sub>,</sub> P<sub>a</sub>tri<sub>c</sub>k S<sub>c</sub>hr<sub>a</sub>m<sub>ows</sub>ki<sub>,</sub> Sri<sub>va</sub>t<sub>sa</sub> R K<sub>u</sub>nd<sub>u</sub>rth<sub>y,</sub> K<sub>a</sub>th<sub>e</sub>rin<sub>e</sub> Cr<sub>owso</sub>n<sub>,</sub> Ludwi<sub>g</sub> Schmidt, Robert Kaczmarcz<sub>y</sub>k, and Jenia Jitsev. 2022. LAION-5B: An

o<sub>p</sub>en lar<sub>g</sub>e-scale dataset for trainin<sub>g</sub> next <sub>g</sub>eneration ima<sub>g</sub>e-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track. https://openreview.net/forum?id=M3Y74vmsMcY

[69] Paul S. Scotti, Atmadee<sub>p</sub> Banerjee, Jimmie Goode, Ste<sub>p</sub>an Shabalin, Alex N<sub>g</sub>u<sub>y</sub>en, Ethan Cohen, Aidan J. Dem<sub>p</sub>ster, Nathalie Verlinde, Elad Yundler, David Weisber<sub>g</sub>, K<sub>e</sub>nn<sub>e</sub>th A<sub>.</sub> N<sub>o</sub>rm<sub>a</sub>n<sub>,</sub> <sub>a</sub>nd T<sub>a</sub>ni<sub>s</sub>h<sub>q</sub> M<sub>a</sub>th<sub>ew</sub> Abr<sub>a</sub>h<sub>a</sub>m<sub>.</sub> 2023<sub>.</sub> R<sub>eco</sub>n<sub>s</sub>tr<sub>uc</sub>tin<sub>g</sub> th<sub>e</sub> Mind’<sub>s</sub> E<sub>ye</sub>: fMRI-t<sub>o</sub>-Im<sub>age w</sub>ith C<sub>o</sub>ntr<sub>as</sub>ti<sub>ve</sub> L<sub>ea</sub>rnin<sub>g a</sub>nd Dif<sub>us</sub>i<sub>o</sub>n Pri<sub>o</sub>r<sub>s.</sub> arXiv:2305.18274 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2305.18274

[70] Paul S. Scotti, Mihir Tri<sub>p</sub>ath<sub>y</sub>, Cesar Kadir Torrico Villanueva, Reese Kneeland, Ton<sub>g</sub> Chen, Ashutosh Naran<sub>g</sub>, Charan Santhirase<sub>g</sub>aran, Jonathan Xu, Th<sub>omas</sub> N<sub>ase</sub>l<sub>ar</sub>i<sub>s,</sub> K<sub>enne</sub>th A<sub>.</sub> N<sub>orman, an</sub>d T<sub>an</sub>i<sub>s</sub>h<sub>q</sub> M<sub>a</sub>th<sub>ew</sub> Ab<sub>ra</sub>h<sub>am.</sub> 2024<sub>.</sub> Min<sup>d</sup>Eye2: S<sup>h</sup>are<sup>d</sup>-Su<sup>b</sup>ject Mo<sup>d</sup>e<sup>l</sup>s Ena<sup>bl</sup>e <sup>f</sup>MRI-To-Image Wit<sup>h</sup> 1 Hour o<sup>f</sup> Data. arXiv:2403.11207 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2403.11207

[71] Guohua Shen, Tomo<sub>y</sub>asu Horikawa, Kei Majima, and Yuki<sub>y</sub>asu Kamitani. 2019. Deep image reconstruction from human brain activity. PLoS Computational Biology 15, 1 (2019), e1006633.

[72] Matthew Shinkle and Mark D. Lescroart. 2025. Visualizin<sub>g</sub> and Controllin<sub>g</sub> Cortical Responses Using Voxel-Weighted Activation Maximization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. 4864–4868.

[73] Yon<sub>g</sub>hao Son<sub>g</sub>, Bin<sub>g</sub>chuan Liu, Xian<sub>g</sub> Li, Nanlin Shi, Yijun Wan<sub>g</sub>, and Xiaorong Gao. 2024. Deco<sup>d</sup>ing Natura<sup>l</sup> Images <sup>f</sup>rom EEG <sup>f</sup>or O<sup>b</sup>ject Recognition. arXiv:2308.13234 [cs.HC] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2308.13234

[74] Christian Sze<sub>g</sub>ed<sub>y</sub>, Wojciech Zaremba, Il<sub>y</sub>a Sutskever, Joan Bruna, Dumitru Erhan<sub>,</sub> Ian Goodfellow<sub>,</sub> and Rob Fer<sub>g</sub>us. 2013. Intri<sub>g</sub>uin<sub>g p</sub>ro<sub>p</sub>erties of neural networks. In arXiv preprint arXiv:1312.6199.

[75] Yu Taka<sub>g</sub>i and Shinji Nishimoto. 2023. Hi<sub>g</sub>h-Resolution Ima<sub>g</sub>e Reconstruction With Latent Difusion Models From Human Brain Activity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 14453–14463.

[76] Mike Tyka. 2016. Regularizing feature visualization with bilateral filters. Personal communication (2016)

[77] Aaron Van den Oord, Nal Kalchbrenner, Lasse Es<sub>p</sub>eholt, Oriol Vin<sub>y</sub>als, Alex Gr<sub>aves, a</sub>nd K<sub>o</sub>r<sub>ay</sub> K<sub>avu</sub>k<sub>cuog</sub>l<sub>u.</sub> 2016<sub>.</sub> C<sub>o</sub>nditi<sub>o</sub>n<sub>a</sub>l Im<sub>age</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>w</sub>ith Pix<sub>e</sub>l CNN Decoders. In Advances in Neural Information Processing Systems (NeurIPS).

[78] Marcel A.J. van Gerven. 2017. A <sub>p</sub>rimer on encodin<sub>g</sub> models in sensor<sub>y</sub> neuroscience. Journal ofMathematical Psychology 76 (2017), 172–183. doi:10.1016/j. jmp.2016.06.009 Mo<sup>d</sup>e<sup>l</sup>-<sup>b</sup>ase<sup>d</sup> Cognitive Neuroscience.

[79] Marcel A J van Gerven. 2021. Neural decoding with generative models. Trends in Cognitive Sciences 25, 7 (2021), 593–607.

[80] Ed<sub>g</sub>ar Walker, Fabian Sinz, Erick Cobos, Taliah Muhammad, Emmanouil Froudarakis, Paul Fahe<sub>y</sub>, Alexander Ecker, Jacob Reimer, Xa<sub>q</sub> Pitkow, and Andreas Tolias. 2019. Ince<sub>p</sub>tion loo<sub>p</sub>s discover what excites neurons most usin<sub>g</sub> dee<sub>p</sub> predictive models. Nature Neuroscience 22 (12 2019), 1–6. doi:10.1038/s41593- 019-0517-x

[81] Shizun Wan<sub>g</sub>, Son<sub>g</sub>hua Liu, Zhenxion<sub>g</sub> Tan, and Xinchao Wan<sub>g</sub>. 2024. Mind-Brid<sub>g</sub>e: A Cross-Subject Brain Decodin<sub>g</sub> Framework. arXiv:2404.07850 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2404.07850

[82] Weihao Xia, Raoul de Charette, Cen<sub>g</sub>iz Öztireli, and Jin<sub>g</sub>-Hao Xue. 2024. DREAM: Visual Decodin<sub>g</sub> from Reversin<sub>g</sub> Human Visual S<sub>y</sub>stem. arXiv:2310.02265 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2310.02265

[83] Hu Ye, Jun Zhan<sub>g</sub>, Sibo Liu, Xiao Han, and Wei Yan<sub>g</sub>. 2023. IP-Ada<sub>p</sub>ter: Text Com<sub>p</sub>atible Ima<sub>g</sub>e Prom<sub>p</sub>t Ada<sub>p</sub>ter for Text-to-Ima<sub>g</sub>e Dif<sub>u</sub>sion Models. arXiv:2308.06721 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2308.06721

[84] Bohan Zen<sub>g</sub>, Shan<sub>g</sub>lin Li, Xuhui Liu, Sichen<sub>g</sub> Gao, Xiaolon<sub>g</sub> Jian<sub>g</sub>, Xu Tan<sub>g</sub>, Yao Hu, Jianzhuan<sub>g</sub> Liu, and Baochan<sub>g</sub> Zhan<sub>g</sub>. 2023. Controllable Mind Visual Difusion Model. arXiv:2305.10135 [cs.CV] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2305.10135

# Supplementary Material

## Appendix Overview

Thi<sub>s appen</sub>di<sub>x prov</sub>id<sub>es compre</sub>h<sub>ens</sub>i<sub>ve</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s an</sub>d <sub>a</sub>dditi<sub>ona</sub>l <sub>resu</sub>lt<sub>s</sub> t<sub>o suppor</sub>t th<sub>e</sub> fi<sub>n</sub>di<sub>ngs presen</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e ma</sub>i<sub>n paper an</sub>d <sub>ensure</sub> th<sub>a</sub>t <sub>a</sub>ll <sub>o</sub>f th<sub>e resu</sub>lt<sub>s can</sub> b<sub>e repro</sub>d<sub>uce</sub>d<sub>.</sub> Th<sub>e appen</sub>di<sub>x con</sub>t<sub>en</sub>t i<sub>s organ</sub>i<sub>ze</sub>d i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng sec</sub>ti<sub>ons:</sub>

• Additional Results: Additiona<sup>l</sup> visua<sup>l</sup>izations and ana<sup>l</sup>yses t<sup>h</sup>at <sup>f</sup>urt<sup>h</sup>er demonstrate t<sup>h</sup>e e<sup>f</sup>ectiveness o<sup>f</sup> our proposed met<sup>h</sup>ods. M<sub>ore examp</sub>l<sub>es o</sub>f <sub>en</sub>h<sub>ance</sub>d <sub>samp</sub>li<sub>ng,</sub> l<sub>oca</sub>l <sub>c</sub>h<sub>ange resu</sub>lt<sub>s are presen</sub>t<sub>e</sub>d<sub>, as s</sub>h<sub>own</sub> i<sub>n sec</sub>ti<sub>on</sub> A<sub>.</sub>

• Experimental Settings: Compre<sup>h</sup>ensive in<sup>f</sup>ormation about t<sup>h</sup>e datasets settings, <sup>l</sup>i<sup>k</sup>e input size and <sup>h</sup>ow are t<sup>h</sup>e datasets divided <sub>an</sub>d <sub>processe</sub>d<sub>, mo</sub>d<sub>e</sub>l <sub>arc</sub>hit<sub>ec</sub>t<sub>ures an</sub>d i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s,</sub> th<sub>en</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng con</sub>fi<sub>gura</sub>ti<sub>ons an</sub>d <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>resources.</sub> All <sub>o</sub>f th<sub>em can prov</sub>id<sub>e a c</sub>l<sub>ear un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng o</sub>f <sub>our exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>se</sub>t<sub>up, w</sub>hi<sub>c</sub>h <sub>are</sub> h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l t<sub>o repro</sub>d<sub>uce</sub> th<sub>e resu</sub>lt<sub>s, as s</sub>h<sub>own</sub> i<sub>n sec</sub>ti<sub>on</sub> B<sub>.</sub>

• Implementation Details: Detai<sup>l</sup>ed in<sup>f</sup>ormation about t<sup>h</sup>e a<sup>l</sup> orit<sup>h</sup>ms we used to im <sup>l</sup>ement our met<sup>h</sup>od, inc<sup>l</sup>udin seudocode, and corres<sub>p</sub>ondin<sub>g</sub> ex<sub>p</sub>lanations. All of them can <sub>p</sub>rovide a clear understandin<sub>g</sub> of how the trainin<sub>g</sub> and o<sub>p</sub>timization are carried out<sub>,</sub> as <sub>s</sub>h<sub>ow</sub>n in <sub>sec</sub>ti<sub>o</sub>n C<sub>.</sub>

• Predictive Power of In-silico brain simulator: In<sup>f</sup>ormation about <sup>h</sup>ow muc<sup>h</sup> our trained In-si<sup>l</sup>ico brain simu<sup>l</sup>ator can predict t<sup>h</sup>e br<sub>a</sub>in <sub>ac</sub>ti<sub>v</sub>it<sub>y</sub> d<sub>a</sub>t<sub>a, as s</sub>h<sub>ow</sub>n in <sub>sec</sub>ti<sub>o</sub>n D<sub>.</sub>

## A Additional Results

## A.1 More Results for the NSD Dataset

Here is a s<sub>u</sub>mmarized table of more ex<sub>p</sub>erimental res<sub>u</sub>lts on NSD dataset:

Table S1: Activation change and realism on additional ROIs. Better values are in bold.
<table><tr><td></td><td>ROIs</td><td>V1</td><td>V2</td><td>V3</td><td>V4</td><td>V8</td><td>LO1</td><td>LO2</td><td>V4t</td></tr><tr><td>Activation Increase (∆r)</td><td>Ours</td><td>0.257</td><td>0.080</td><td>0.247</td><td>0.486</td><td>0.696</td><td>0.583</td><td>0.498</td><td>0.757</td></tr><tr><td rowspan="2">Activation</td><td>BrainACTIV</td><td>0.157</td><td>0.116</td><td>0.226</td><td>0.488</td><td>0.721</td><td>0.582</td><td>0.517</td><td>0.767</td></tr><tr><td>Ours</td><td>-0.291</td><td>-0.272</td><td>-0.573</td><td>-0.690</td><td>-0.406</td><td>-0.485</td><td>-0.490</td><td>-0.601</td></tr><tr><td>Decrease (∆r)</td><td>BrainACTIV</td><td>0.102</td><td>0.224</td><td>0.274</td><td>0.501</td><td>0.608</td><td>0.100</td><td>-0.057</td><td>-0.511</td></tr><tr><td rowspan="2">FID@2k</td><td>Ours</td><td>164.87</td><td>157.84</td><td>167.44</td><td>173.24</td><td>160.07</td><td>136.05</td><td>186.67</td><td>178.62</td></tr><tr><td>BrainACTIV</td><td>224.29</td><td>201.56</td><td>240.64</td><td>231.60</td><td>216.80</td><td>208.23</td><td>255.06</td><td>197.61</td></tr><tr><td></td><td>ROIs</td><td>VWFA-1</td><td>PPA</td><td>RSC</td><td>OWFA</td><td>IP2</td><td>TPOJ1</td><td>TPOJ2</td><td>TPOJ3</td></tr><tr><td rowspan="2">Activation Increase (∆r)</td><td>Ours</td><td>1.108</td><td>0.952</td><td>0.923</td><td>0.718</td><td>0.091</td><td>0.456</td><td>0.661</td><td>1.014</td></tr><tr><td>BrainACTIV</td><td>1.019</td><td>0.926</td><td>0.954</td><td>1.088</td><td>0.084</td><td>0.509</td><td>0.751</td><td>1.091</td></tr><tr><td rowspan="2">Activation Decrease (∆r)</td><td>Ours</td><td>-0.391</td><td>-0.809</td><td>-0.913</td><td>-0.780</td><td>-0.213</td><td>-0.136</td><td>-0.384</td><td>-0.591</td></tr><tr><td>BrainACTIV</td><td>0.350</td><td>-0.579</td><td>-0.804</td><td>0.347</td><td>0.061</td><td>-0.134</td><td>-0.234</td><td>-0.484</td></tr><tr><td rowspan="2">FID@2k</td><td>Ours</td><td>169.14</td><td>153.67</td><td>172.92</td><td>163.88</td><td>168.56</td><td>192.04</td><td>131.83</td><td>137.57</td></tr><tr><td>BrainACTIV</td><td>229.65</td><td>234.79</td><td>216.65</td><td>207.15</td><td>171.94</td><td>244.44</td><td>197.94</td><td>197.24</td></tr></table>

The f<sub>u</sub>ll test-set res<sub>u</sub>lts for the additional ROIs are <sub>p</sub>rovided in Fi<sub>g</sub>s. S1–S16.

![](images/a95a62c9ef6b9e4c2e572c16db5a0ef518ed563aaff39b142943596004ed6d2d.jpg)  
Figure S1: Full result of activation change on region IPS4, positive direction.

Original

![](images/38e793e3823155d1a3fd4ed6865b14271ec994e6673a7c9a64c0916730a7f606.jpg)  
Figure S2: Full result of activation change on region IPS4, negative direction.

![](images/4aac40ca1ed8ae24ef5119ff520f4f4fe02c050e2f83e3627e7574d6888d8d41.jpg)  
Figure S3: Full result of activation change on region SPL1, positive direction.

Original

Original  
![](images/e008b5c40ce5623b084bc4dc0a70f5a385fe522b94addc85b8deff48297ee20d.jpg)  
Figure S4: Full result of activation change on region SPL1, negative direction.

![](images/547a78c5ef7e0e4e2678770e4c6190792fac842d1ce14e71ea1da6657ff72a43.jpg)  
Figure S5: Full result of activation change on region 31a, positive direction.

� = −�. �

Original  
Original  
![](images/79dae87aa2d20cf35d50fd14f326e1f15a42b1c1b00f6d820488b8775ecfccbc.jpg)  
Figure S6: Full result of activation change on region 31a, negative direction.

![](images/adc65cf4f25915a79bb60c2dd007a55d379954d7e16209dff2cfdc727c0cd3b7.jpg)  
Figure S7: Full result of activation change on region 31pd, positive direction.

Original  
Original  
![](images/c79f2d9fe457ea3626cdfd76e3ccde8afefca28fef04f472026d3dca6362d166.jpg)  
Figure S8: Full result of activation change on region 31pd, negative direction.

![](images/f56f89b5248666fd84f59b569430bfe5ce496603fc54c9c5cc8bbb2f2a0d2e42.jpg)  
Figure S9: Full result of activation change on region V1, positive direction.

Original  
Original  
![](images/fc870227b3f3a1c04242e3733dbf8f0628828895fbd78cb126e25df225cb031c.jpg)  
Figure S10: Full result of activation change on region V1, negative direction.

Original

Original

![](images/61c7e1b02cc0b863aff3b620d1b86d2266e2c764b7e79afcff03d2b1820de878.jpg)  
Figure S11: Full result of activation change on region V2, positive direction.

Original  
Original  
![](images/b94497706392cf918c2001cb1ac5e8e805ffca8275bc4871c0fe79453fbab38d.jpg)  
Figure S12: Full result of activation change on region V2, negative direction.

Original

Original

![](images/a47f4a015c8ec2a445a77f28c6d318aba6ce404349e34cabe7c8b24cf5ad741a.jpg)  
Figure S13: Full result of activation change on region V3, positive direction.

Original

![](images/1a94f3079c9e2e6b203ee3ee232b4028086902e296b2cd97ea8245f9f622e9ba.jpg)  
Figure S14: Full result of activation change on region V3, negative direction.

Original  
Original  
![](images/5565d39968851b58f3f045e6927184c4685fdab6bb61aa5583644d060eb0deee.jpg)  
� = �. �  
Figure S15: Full result of activation change on region V4, positive direction.

![](images/c9e6b2da27dcbc9393ae25473f1895d8a836f2a060aec196a5b3309725b1d573.jpg)  
Figure S16: Full result of activation change on region V4, negative direction.

## A.2 More ANN Results

We <sub>p</sub>resent more exam<sub>p</sub>les on mani<sub>p</sub>ulatin<sub>g</sub> activations in SimCLR[8], shown in Fi<sub>g</sub>s. S17–S22, which further demonstrate the efectiveness <sub>o</sub>f <sub>our me</sub>th<sub>o</sub>d<sub>.</sub>

Top 10 Images  
![](images/4d44817326f8da4ffb0001df5281901b2aa6f37cfbc3159f8f0e1fbf46c6aa8e.jpg)  
Figure S17: One latent modification example to generate preferred images with increasing unit activations on ImageNet-mini Dataset. Though the top 10 images can not demonstrate a notable pattern represented by the unit in the early layer of SimCLR model, our method can generate a uniform representation and compared with original images from ImageNet-mini, generated preferred images based on modifying the latent to the positive prior mean shows a more apparent change in activation compared with original images and representations from BrainACTIV.

Top 10 Images  
![](images/64f9c77697932eaac4b685ff992cbda10a4600f5b308ac1f59fe7c7091390d75.jpg)  
Figure S18: One latent modification example to generate preferred images with increasing unit activations on ImageNet-mini Dataset. Though the top 10 images can not demonstrate a notable pattern represented by the unit in the early layer of SimCLR model, our method can generate a uniform representation and compared with original images from ImageNet-mini, generated preferred images based on modifying the latent to the positive prior mean shows a more apparent change in activation compared with original images and representations from BrainACTIV.

![](images/fe3be57f61da64d78cce019bd6fe42841543c8e6d460117c78f1576415d53933.jpg)

Top 10 Images  
![](images/64036dade10793d83f142dd6f4230850fed4878f87b8ce32455fa7b5350ccd49.jpg)  
Figure S19: One latent modification example to generate preferred images with increasing unit activations on ImageNet-mini Dataset. Though the top 10 images can not demonstrate a notable pattern represented by the unit in the early layer of SimCLR model, our method can generate a uniform representation and compared with original images from ImageNet-mini, generated preferred images based on modifying the latent to the positive prior mean shows a more apparent change in activation compared with original images and representations from BrainACTIV.  
Top 10 Images

![](images/806cb24340201927276319dd80b01701e9e21a04f5a24c5ce90b0b7066971530.jpg)  
Figure S20: One latent modification example to generate preferred images with increasing unit activations on ImageNet-mini Dataset. The top 10 images demonstrate a notable pattern represented by the unit in the deep layer of SimCLR model, that round objects appear in top images. Our method can also generate a uniform representation and compared with original images from ImageNet-mini, generated preferred images based on modifying the latent to the positive prior mean shows a more apparent change in activation compared with original images and representations from BrainACTIV.

Top 10 Images  
![](images/a8fc2f511d46b5491cc4131f68b40ecadbc1680aa97ceb8deb4f4e48940ec6b5.jpg)  
Figure S21: One latent modification example to generate preferred images with increasing unit activations on ImageNet-mini Dataset. The top 10 images demonstrate a notable pattern represented by the unit in the deep layer of SimCLR model, that grid objects appear in top images. Our method can also generate a uniform representation and compared with original images from ImageNet-mini, generated preferred images based on modifying the latent to the positive prior mean shows a more apparent change in activation compared with original images and representations from BrainACTIV.

Top 10 Images  
![](images/361250812de53984ea57aa79530884a24b6112c8e30eed954a5113886a605c47.jpg)  
Figure S22: One latent modification example to generate preferred images with increasing unit activations on ImageNet-mini Dataset. Though the top 10 images can not demonstrate a notable pattern represented by the unit in the early layer of SimCLR model, our method can generate a uniform representation and compared with original images from ImageNet-mini, generated preferred images based on modifying the latent to the positive prior mean shows a more apparent change in activation compared with original images and representations from BrainACTIV.

## B Experimental Settings

## B.1 Dataset Details

W<sub>e w</sub>ill <sub>s</sub>h<sub>ow</sub> h<sub>ow</sub> th<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t <sub>was processe</sub>d i<sub>n</sub> thi<sub>s par</sub>t<sub>.</sub>

Natural Scenes Dataset (NSD): [2]: we se<sup>l</sup>ected 1000 images presented to Subject 1 <sup>f</sup>rom t<sup>h</sup>e dataset as our test set, w<sup>h</sup>i<sup>l</sup>e using CLIP em<sup>b</sup>e<sup>dd</sup>ings o<sup>f</sup> 9000 images s<sup>h</sup>own to same su<sup>b</sup>ject <sup>f</sup>or training. We use<sup>d</sup> same test images se<sup>l</sup>ecte<sup>d</sup> <sup>b</sup>y t<sup>h</sup>e BrainACTIV. An<sup>d</sup> <sup>f</sup>or t<sup>h</sup>e measurement of activation chan<sub>g</sub>e and FID score, we use all 1000 ima<sub>g</sub>es <sub>p</sub>resented to same subject as test set in COCO dataset [34] and their CLIP embeddin<sub>g</sub>s. 2k ima<sub>g</sub>es for FID[22] calculation are the combination of 1k <sub>p</sub>referred ima<sub>g</sub>es to maximize/minimize the activations, res<sub>p</sub>ectivel<sub>y</sub>. The CLIP embeddin<sub>g</sub>s are retrieved from the first out<sub>p</sub>ut of CLIP encoder [64], the dimension is 1024 for ever<sub>y</sub> ima<sub>g</sub>e.

Imagenet-mini [13]: T<sup>h</sup>e train and test sp<sup>l</sup>it <sup>f</sup>o<sup>ll</sup>ow t<sup>h</sup>e de<sup>f</sup>au<sup>l</sup>t settings in t<sup>h</sup>e orgina<sup>l</sup> dataset, w<sup>h</sup>ic<sup>h</sup> divided t<sup>h</sup>e w<sup>h</sup>o<sup>l</sup>e dataset into tr<sub>a</sub>in/<sub>va</sub>l <sub>se</sub>t<sub>s, we use</sub> tr<sub>a</sub>in <sub>as ou</sub>r tr<sub>a</sub>in <sub>se</sub>t <sub>a</sub>nd <sub>va</sub>l <sub>as ou</sub>r t<sub>es</sub>t<sub>se</sub>t<sub>.</sub> Tr<sub>a</sub>in <sub>se</sub>t <sub>co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>s o</sub>f 34<sub>,</sub>745 im<sub>ages, a</sub>nd t<sub>es</sub>t <sub>se</sub>t <sub>co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>s o</sub>f 3<sub>,</sub>923 im<sub>ages.</sub> F<sub>o</sub>r the trainin<sub>g p</sub>rocess of SimCLR<sub>,</sub> we used followin<sub>g</sub> transform techni<sub>q</sub>ues in torchvision.transforms to edit the or<sub>g</sub>inal Ima<sub>g</sub>enet-mini ima<sub>g</sub>es:

t r a n s fo r m s . RandomResizedCro<sub>p</sub> ( 1 2 8 , s c a l e = ( 0 . 0 8 , 1 . 0 ) ,

i n t e r <sub>p</sub> o l a t i o n = I m a <sub>g</sub> e . BICUBIC )

t r a n s f o r m s . R a n d o m H o r i z o n t a l F l i <sub>p</sub> ( )

t r a n s f o r m s . T o T e n s o r ( )

## B.2 Model Implementation Details

## B.2.1 VAE Implementation and Training.

VAE:. For NSD and ImageNet dataset to <sup>l</sup>earn CLIP embeddings, using VAE arc<sup>h</sup>itecture [30] consists o<sup>f</sup> an encoder and a decoder wit<sup>h</sup> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> l<sub>ayers:</sub>

• Encoder (CLIP → latent)

– FC : 1024 −→ 4096 (first 2048 dims are <sub>�</sub>, last 2048 dims are lo<sub>g</sub> �<sup>2</sup>)

• Latent layer

– <sub>�</sub> ∈ R<sup>2048</sup>, lo<sub>g</sub> �<sup>2</sup> ∈ R<sup>2048</sup>

– Re<sub>p</sub>arameterisation: � = <sub>�</sub> + � ⊙ ex<sub>p</sub>(0.5 lo<sub>g</sub> �<sup>2</sup>)

• Decoder (latent → CLIP)

$$
- \ { \mathrm { F C } } _ { \mathrm { d e c } } \colon 2 0 4 8 \longrightarrow 1 0 2 4 ( { \mathrm { r e c o n s t r u c t e d } } \ { \mathrm { C L I P ~ e m b e d d i n g } } )
$$

Thi<sub>s</sub> CLIP-t<sub>o</sub>-CLIP VAE i<sub>s</sub> di<sub>s</sub>tin<sub>c</sub>t fr<sub>o</sub>m th<sub>e</sub> im<sub>age</sub>-<sub>space</sub> VAE <sub>use</sub>d in th<sub>e</sub> l<sub>a</sub>t<sub>e</sub>nt <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s, w</sub>h<sub>ose</sub> d<sub>eco</sub>d<sub>e</sub>r <sub>p</sub>r<sub>o</sub>d<sub>uces</sub> im<sub>ages</sub> di<sub>rec</sub>tl<sub>y.</sub>

Th<sub>e</sub> VAE tr<sub>a</sub>inin<sub>g</sub> <sub>co</sub>nfi<sub>gu</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> <sub>a</sub>r<sub>e</sub> <sub>s</sub>h<sub>ow</sub>n in th<sub>e</sub> T<sub>a</sub>bl<sub>e</sub> S2<sub>.</sub>

B.2.2 Implementation ofDifusion Pipeline. All images are first converted to a 1024-dimensional representation with the frozen CLIP-ViT H/14 encoder released b<sub>y</sub> LAION [68]. For ever<sub>y</sub> s<sub>p</sub>lit of the Natural Scenes Dataset we iterate once over the raw-<sub>p</sub>ixel loader, com<sub>p</sub>ute the embeddin<sub>g</sub>s and cache the resultin<sub>g</sub> CLIP features. Subse<sub>q</sub>uent ex<sub>p</sub>eriments therefore o<sub>p</sub>erate <sub>p</sub>urel<sub>y</sub> in embeddin<sub>g</sub> s<sub>p</sub>ace and never revisit th<sub>e expens</sub>i<sub>ve v</sub>i<sub>s</sub>i<sub>on</sub> b<sub>ac</sub>kb<sub>one.</sub>

Ima e s nthesis uses the “im -to-im ” variant of Stable Difusion v1.5 from the Diffusers librar [31]. We swa the ori inal PNDM scheduler for DDIM (better s<sub>p</sub>eed/<sub>q</sub>ualit<sub>y</sub> trade-of) and disable the safet<sub>y</sub> checker to avoid unintended filterin<sub>g</sub>. Ever<sub>y</sub> network in the <sub>p</sub>i<sub>p</sub>eline—U-Net, text encoder, VAE, and the IP-Ada<sub>p</sub>ter [83] that injects the CLIP embeddin<sub>g</sub> into the cross-attention blocks—remains entirel<sub>y</sub> frozen. We use an IP-Ada<sub>p</sub>ter check<sub>p</sub>oint com<sub>p</sub>atible with the ViT-H/14 ima<sub>g</sub>e embeddin<sub>g</sub>s em<sub>p</sub>lo<sub>y</sub>ed for feature extraction.

Durin<sub>g</sub> inference each RGB frame (512 × 512 <sub>p</sub>x) is su<sub>pp</sub>lied to<sub>g</sub>ether with a tar<sub>g</sub>et CLIP embeddin<sub>g</sub>—either the baseline embeddin<sub>g</sub> obtained b<sub>y</sub> rid<sub>g</sub>e re<sub>g</sub>ression or the one <sub>p</sub>roduced b<sub>y</sub> our activation-aware VAE. Unless noted otherwise we set the SDEdit stren<sub>g</sub>th to $\gamma = 1 . 0 ;$ We run 50 DDIM ste<sub>p</sub>s <sub>p</sub>er ima<sub>g</sub>e<sub>,</sub> draw a sin<sub>g</sub>le sam<sub>p</sub>le<sub>,</sub> and fix the random seed to 42 for strict com<sub>p</sub>arabilit<sub>y</sub> across all methods

## B.2.3 Implementation ofOther Models.

SimCLR. We train a SimCLR encoder with a ResNet-50 backbone on ImageNet-mini, as described in Section 4.3.2. Training uses the 128 × 128 ima<sub>g</sub>e cro<sub>p</sub>s and au<sub>g</sub>mentations s<sub>p</sub>ecified above, with the model adjusted to accommodate this in<sub>p</sub>ut resolution.

## B.3 Metric Details

Quantifying Similarity. To further evaluate the structured landscape, we used similarity and realism metrics. For similarity between ori<sub>g</sub>inal ima<sub>g</sub>es and ima<sub>g</sub>es with activation o<sub>p</sub>timized but unrelated features <sub>p</sub>reserved<sub>,</sub> we use cosine similarit<sub>y</sub> as follows. Given a set of ima<sub>g</sub>es {� } with feature embeddin<sub>g</sub>s {f }, the similarit<sub>y</sub> score is ex<sub>p</sub>ressed as:

$$
{ \mathrm { C o s i n e ~ S i m i l a r i t y } } ( \mathbf { f } _ { i } , \mathbf { f } _ { j } ) = { \frac { \mathbf { f } _ { i } \cdot \mathbf { f } _ { j } } { \| \mathbf { f } _ { i } \| \| \mathbf { f } _ { j } \| } }
$$

Implementation of Similarity Calculation: Following Section 4.3.3, feature preservation is quantified using a pretrained ResNet18 encoder [21]. We com<sub>p</sub>ute cosine similarit<sub>y</sub> between the feature embeddin<sub>g</sub> of each ori<sub>g</sub>inal ima<sub>g</sub>e and that of its corres<sub>p</sub>ondin<sub>g</sub> o<sub>p</sub>timized ima<sub>g</sub>e. Hi<sub>g</sub>her similarit<sub>y</sub> indicates <sub>g</sub>reater <sub>p</sub>reservation of ima<sub>g</sub>e features.

Table S2: VAE Hyperparameters and Training Settings for NSD and ImageNet-mini datasets.
<table><tr><td>Hyperparameter</td><td>NSD</td><td>ImageNet</td></tr><tr><td colspan="3">Data Hyperparameters</td></tr><tr><td>input_size</td><td>1024 dim embeddings</td><td>1024 dim embeddings</td></tr><tr><td colspan="3">n_channels</td></tr><tr><td>Model Hyperparameters</td><td></td><td></td></tr><tr><td colspan="3">latent_dim 2048</td></tr><tr><td>Training Settings</td><td></td><td></td></tr><tr><td colspan="3">0</td></tr><tr><td>seed batch_size</td><td>128</td><td>0</td></tr><tr><td>epochs</td><td></td><td>128 150</td></tr><tr><td>optimizer</td><td>150 Adam</td><td>Adam</td></tr><tr><td>lr</td><td>1 × 10−4</td><td>1 × 10−3</td></tr><tr><td>scheduler</td><td>StepLR (step_size=10, γ = 0.9)</td><td>StepLR (step_size=10, γ = 0.9)</td></tr><tr><td>grad_clip</td><td></td><td></td></tr><tr><td></td><td>1.0</td><td>1.0</td></tr></table>

Quantifying Realism: Frechét Inception Distance (FID):. To quantify overall realism we report the Frechét Inception Distance, computed on the 2 048-D <sub>p</sub>ool-3 activations of a frozen Ince<sub>p</sub>tion-v3 network. Let $\mu _ { r } , \Sigma _ { r }$ <sub>an</sub>d $\mu _ { g } , \Sigma _ { g }$ b<sub>e</sub> th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>means an</sub>d <sub>covar</sub>i<sub>ances o</sub>f th<sub>e rea</sub>l <sub>an</sub>d <sub>g</sub>enerated ima<sub>g</sub>e features<sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>. The FID is

$$
\begin{array} { r } { \mathrm { F I D } = \Vert \mu _ { r } - \mu _ { g } \Vert _ { 2 } ^ { 2 } + \operatorname { T r } \bigl ( \Sigma _ { r } + \Sigma _ { g } - 2 \left( \Sigma _ { r } \Sigma _ { g } \right) ^ { 1 / 2 } \bigr ) . } \end{array}
$$

Lower values indicate that the <sub>g</sub>enerated distribution is closer to the real one. For the NSD evaluation<sub>,</sub> each method <sub>g</sub>enerates 1<sub>,</sub>000 ima<sub>g</sub>es in the <sub>p</sub>ositive direction and 1<sub>,</sub>000 in the ne<sub>g</sub>ative direction. We <sub>p</sub>ool these into a sin<sub>g</sub>le set of 2<sub>,</sub>000 <sub>g</sub>enerated ima<sub>g</sub>es <sub>p</sub>er method and com<sub>p</sub>ute one FID a<sub>g</sub>ainst the 1,000 ori<sub>g</sub>inal test ima<sub>g</sub>es re<sub>p</sub>eated twice. FID is com<sub>p</sub>uted with the torchmetrics im<sub>p</sub>lementation usin<sub>g</sub> feature=2048.

## B.4 Computational Resources

A<sup>ll</sup> experiments were carried out on a sing<sup>l</sup>e wor<sup>k</sup>station equipped wit<sup>h</sup> 8 × NVIDIA RTX A5000 GPUs (eac<sup>h</sup> wit<sup>h</sup> 24 GB on-board memory, CUDA 12.2, driver 535.183), an AMD EPYC 7543 CPU (32 cores, 2.8 GHz) and 512 GB of s<sub>y</sub>stem RAM. One GPU was used for model trainin<sub>g</sub> or inference at a time. Trainin<sub>g</sub> a CLIP-to-CLIP VAE on the NSD train s<sub>p</sub>lit (≈ 9 k ima<sub>g</sub>es) with a batch-size of 128 took ≈ 6.5 hours on a sin<sub>g</sub>le A5000; O<sub>p</sub>timizin<sub>g</sub> on Ima<sub>g</sub>enet-mini com<sub>p</sub>leted in under 40 minutes. For difusion-based <sub>g</sub>eneration (IP-Ada<sub>p</sub>ter + Stable-Difusion v1.5, 50 DDIM ste<sub>p</sub>s) ,<sub>p</sub>roducin<sub>g</sub> one full 2 k-ima<sub>g</sub>e evaluation set (<sub>p</sub>ositives + ne<sub>g</sub>atives) took about 70 minutes.

## C Implementation Details

## C.1 Training Details

T<sup>h</sup>e <sup>f</sup>o<sup>ll</sup>owing pseu<sup>d</sup>oco<sup>d</sup>e summarizes t<sup>h</sup>e training o<sup>b</sup>jective an<sup>d</sup> generation proce<sup>d</sup>ure <sup>d</sup>escri<sup>b</sup>e<sup>d</sup> in Section 3.3.

Algorithm 1 VAE Training wit<sup>h</sup> Activation Regu<sup>l</sup>arization   
Require:   
1 vae: Variational Autoencoder mode   
2: optimizer: O<sub>p</sub>timizer (Adam)   
3: scheduler: Learnin<sub>g</sub> rate scheduler (Ste<sub>p</sub>LR)   
4: num\_epochs: Number of trainin<sub>g</sub> e<sub>p</sub>ochs   
Ensure:   
5<sub>:</sub> VAE with structured latent s<sub>p</sub>ace and hi<sub>g</sub>h-<sub>q</sub>ualit<sub>y</sub> reconstruction   
6: procedure TrainVAE(vae, optimizer, . .. )   
7: for e oc<sup>h</sup> = 1 to num\_epochs do   
8: for eac<sup>h</sup> batc<sup>h</sup> do   
9: optimizer.zero\_grad()   
10: Forward Pass:   
11: (<sub>�</sub>, lo<sub>g</sub> �<sup>2</sup>) ← vae.encode(data)   
12: <sub>�</sub> ∼ N (0, �); � ← <sub>�</sub> + ex<sub>p</sub>( <sup>1</sup> lo<sub>g</sub> �<sup>2</sup>) ⊙ <sub>�</sub>   
13: �ˆ ← vae.decode(�)   
14: Compute Loss:   
15: R<sub>eco</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n L<sub>oss</sub> $ \| \mathsf { d a t a } - \hat { x } \| _ { 2 } ^ { 2 }$   
16: Compute Prior Parameters:   
17: $\sigma ^ { \mathrm { p r i o r } } \gets \left| \frac { \lambda } { \mathsf { a c t i v a t i o n \_ v a l u e s } + \epsilon } \right|$   
$\mu _ { i } ^ { \mathrm { p r i o r } }  \Bigg \{ \mu _ { \mathrm { p o s } } ,$ if activation\_value $s _ { i } > 0 ,$   
18:   
<sub>o</sub>th<sub>erw</sub>i<sub>se.</sub>   
19: C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub> KLD L<sub>oss</sub> fr<sub>o</sub>m $\mu , \sigma , \mu ^ { \mathrm { p r i o r } } , \sigma ^ { \mathrm { p r i o r } }$ us<sup>i</sup>n<sub>g</sub> <sup>E</sup><sub>q</sub>. 3   
20: Total Loss ← � × Reconstruction Loss $+ \omega _ { \mathrm { K L D } } \times \mathrm { K L D }$ L<sub>oss</sub>   
21: Backpropagation:   
22: Total Loss.backward()   
23: optimizer.step()   
24: end for   
25: scheduler.step()   
26: end for   
27: end procedure

Latent Modification and Image Generation. To generate high-quality preferred images that enhance activations based on the latent <sub>mo</sub>difi<sub>ca</sub>ti<sub>on,</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m</sub> i<sub>s s</sub>h<sub>own as</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> bl<sub>oc</sub>k 2<sub>:</sub>

Algorithm 2 Vis-Lens: Pre<sup>f</sup>erred–Ima e Generation   
Require:   
$i _ { \mathrm { i n } } \ldots$ RGB in<sub>pu</sub>t im<sub>age</sub>   
clip\_enc $\dots$ fr<sub>o</sub>z<sub>e</sub>n CLIP–ViT-H <sub>e</sub>n<sub>co</sub>d<sub>e</sub>r   
$x \cdot \cdot \cdot$ 1024-d CLIP embeddin<sub>g</sub> of $i _ { \mathrm { i n } }$   
vae . . . activation–aware CLIP→CLIP VAE   
$z , \mu \ldots$ latent code / mean<sub>;</sub> <sub>� os,</sub> <sub>�ne</sub> . . . acti<sub>v</sub>ation <sub>p</sub>riors   
ip\_adapter ... frozen IP-Ada<sub>p</sub>ter   
sd $1 5 \ldots$ Stable-Difusion v1.5 (DDIM)   
$\alpha \cdot \cdot \cdot$ linear inter<sub>p</sub>olation wei<sub>g</sub>ht<sub>,</sub> $\gamma \ldots$ SDEdit <sub>s</sub>tr<sub>e</sub>n th   
$r _ { \mathrm { d i r } } \in$ {<sub>p</sub>os, ne<sub>g</sub>} . . . desired ROI shift   
Ensure: pre<sup>f</sup>erred image �<sub>out</sub>   
1: procedure GenPreferre $\mathrm { > } ( \mathrm { i n } , r _ { \mathrm { d i r } } )$   
2: � ← clip\_enc $( i _ { \mathrm { i n } } )$   
3: (<sub>�</sub>, lo<sub>g</sub> �<sup>2</sup>) ← vae.encode(�); � ←<sub>�</sub>   
4: if $r _ { \mathrm { d i r } } = 1$ pos then   
5: �<sub>p</sub>rior $ \mu _ { \mathrm { p o s } }$   
6: else   
7: <sub>�prior</sub> ← <sub>�neg</sub>   
8: end if   
9: $z ^ { \prime } \gets \alpha \mu _ { \mathrm { p r i o r } } + ( 1 - \alpha ) z$   
10: $x _ { \mathrm { g e n } } $ vae.decode(�<sup>′</sup>)   
11: �<sub>out</sub> ← ip\_adapter.generate(cli<sub>p</sub> $= x _ { \mathrm { g e n } } ,$ ima<sub>g</sub>e = �<sub>in</sub>, <sub>�</sub>, 50 ste<sub>p</sub>s)   
12: return �<sub>out</sub>   
13: end procedure

Here are more detailed ex<sub>p</sub>lanations of our <sub>g</sub>eneration <sub>p</sub>rocess:

• The in<sub>p</sub>ut ima<sub>g</sub>e $i _ { \mathrm { i n } }$ is projecte<sup>d</sup> to a CLIP em<sup>b</sup>e<sup>dd</sup>ing x wit<sup>h</sup> t<sup>h</sup>e <sup>f</sup>rozen enco<sup>d</sup>er.

• The activation-aware VAE encodes x to latent mean $\mu ;$ <sub>we</sub> t<sub>a</sub>k<sub>e</sub> $z = \mu$ f<sub>o</sub>r <sub>a</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c e</sub>dit<sub>.</sub>

• Followin<sub>g</sub> Section 4.3.1, the latent edit is $z ^ { \prime } = \alpha \mu _ { \mathrm { p r i o r } } + ( 1 - \alpha ) z$ <sub>, w</sub>h<sub>ere</sub> $\mu _ { \mathrm { p r i o r } }$ i<sub>s</sub> th<sub>e pos</sub>iti<sub>ve p</sub>ri<sub>o</sub>r $\mu _ { \mathrm { p o s } }$ t<sub>o</sub> r<sub>a</sub>i<sub>se ac</sub>ti<sub>v</sub>it<sub>y o</sub>r th<sub>e</sub> n<sub>ega</sub>ti<sub>ve</sub> <sub>p</sub>r<sup>i</sup>or $\mu _ { \mathrm { n { e g } } }$ <sup>to</sup> <sup>su</sup>pp<sup>ress</sup> <sup>it.</sup>

• Decodin<sub>g</sub> z’ <sub>g</sub>ives an intermediate CLIP embeddin<sub>g</sub> $x _ { \mathrm { g e n } }$

$x _ { \mathrm { g e n } }$ is injecte<sup>d</sup>—via a <sup>f</sup>rozen IP-A<sup>d</sup>apter—into Sta<sup>bl</sup>e-Di<sup>f</sup>usion v1.5. Wit<sup>h</sup> SDE<sup>d</sup>it strengt<sup>h</sup> $\gamma = 1 . 0$ th<sub>e</sub> r<sub>eve</sub>r<sub>se</sub> DDIM <sub>p</sub>r<sub>ocess</sub> r<sub>e</sub>li<sub>es</sub> entirel<sub>y</sub> on $x _ { \mathrm { g e n } } ;$ <sub>sma</sub>ll<sub>er</sub> $\gamma$ bl<sub>en</sub>d<sub>s</sub> i<sub>n</sub> l<sub>ow-</sub>l<sub>eve</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> f<sub>rom</sub> th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> i<sub>mage.</sub>

• The sam<sub>p</sub>ler returns the final <sub>p</sub>referred ima<sub>g</sub>e $i _ { \mathrm { o u t } } .$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>use</sub>d i<sub>n</sub> <sub>a</sub>ll d<sub>owns</sub>t<sub>ream</sub> <sub>ana</sub>l<sub>yses.</sub>

## C.2 Details about Optimizing Unit Activations with Latent

The al<sub>g</sub>orithm for o<sub>p</sub>timizin<sub>g</sub> ne<sub>u</sub>ral net<sub>w</sub>ork <sub>u</sub>nit res<sub>p</sub>onses <sub>u</sub>sin<sub>g</sub> latent <sub>v</sub>ariables from a VAE is listed as follo<sub>w</sub>in<sub>g</sub> al<sub>g</sub>orithm block 3:

Structured Visual Landscape: Generating Preferred Representations in Biological and Artificial Neural Networks

Algorithm 3 O timize Activation wit<sup>h</sup> Latent Variab<sup>l</sup>es   
Require:   
1: net: Pre-trained neural network model   
2: vae: Pre-trained Variational Autoencoder   
3: layer\_name: Tar<sub>g</sub>et la<sub>y</sub>er in net   
4: unit\_idx: Index of tar<sub>g</sub>et unit   
5: num\_iterations: Number of o<sub>p</sub>timization ste<sub>p</sub>s   
6: learning\_rate: Learnin<sub>g</sub> rate for o<sub>p</sub>timizer   
7: KLD\_weight: Wei<sub>g</sub>ht for Kullback-Leibler Diver<sub>g</sub>ence loss   
8: target: Direction <sup>“</sup>max" or <sup>“</sup>min")   
9: init\_data: Initial data sam<sub>p</sub>le   
10: J: Latent dimensions associated with the tar<sub>g</sub>et unit   
Ensure:   
11<sub>:</sub> O<sub>p</sub>timized res<sub>p</sub>onses scores<sub>,</sub> Controlled KLD loss   
12: procedure OptimizeActivation(net, vae, . . .)   
13: Initialize Latent Variable:   
14: ( , lo var) ← vae.encode(init\_data)   
15: � ← <sub>�</sub>; �<sub>0</sub> ← <sub>�</sub>   
16: Set Up Optimizer:   
17: optimizer ← Adam(�, lr=learning\_rate)   
18: Register Activation Hook on layer\_name   
19: for iteration = 1 to num\_iterations do   
20: Decode Latent to Image:   
21: � ← vae.decode(�)   
22: Forward Pass:   
23: net(x)   
24: Compute Loss:   
25: if target = <sup>“</sup>max" then   
26: loss ← −activation + �<sub>KLD</sub> × KLD(�)   
27: else   
28: loss ← activation + �<sub>KLD</sub> × KLD(�)   
29: end if   
30: Backpropagation:   
31: o<sub>p</sub>timizer.zero\_<sub>g</sub>rad()   
32: loss.backward()   
33: o<sub>p</sub>timizer.ste<sub>p</sub>()   
34: �<sub>�</sub> ← (�<sub>0</sub> )<sub>�</sub> for all � ∉ J   
35: end for   
36: Remove Activation Hook   
37: Return �, activation scores   
38: end procedure

More Explanation ofthe Algorithm. This algorithm describes the image-space VAE optimization experiment in Section 4.3.3. To preserve unrelated features, onl<sub>y</sub> the latent dimensions J associated with the selected unit are u<sub>p</sub>dated; the remainin<sub>g</sub> coordinates sta<sub>y</sub> at their initia <sub>va</sub>l<sub>ues.</sub> Th<sub>e</sub> VAE <sub>an</sub>d t<sub>arge</sub>t <sub>ne</sub>t<sub>wor</sub>k <sub>rema</sub>i<sub>n</sub> fi<sub>xe</sub>d <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>co</sub>d<sub>e</sub> i<sub>s</sub> <sub>op</sub>ti<sub>m</sub>i<sub>ze</sub>d t<sub>o</sub> i<sub>ncrease</sub> <sub>or</sub> d<sub>ecrease</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>response.</sub>

Initialization: We initia<sup>l</sup>ize t<sup>h</sup>e <sup>l</sup>atent variab<sup>l</sup>e � by encoding t<sup>h</sup>e input image using t<sup>h</sup>e VAE encoder:

$$
( \mu , \mathrm { l o g v a r } ) \gets \mathsf { v a e . e n c o d e ( i n i t \_ d a t a ) }
$$

$$
z \gets \mu
$$

Optimization Loop: For eac<sup>h</sup> iteration, we per<sup>f</sup>orm t<sup>h</sup>e <sup>f</sup>o<sup>ll</sup>owing steps:

1. Decode Latent to Image: T<sup>h</sup>e current <sup>l</sup>atent variab<sup>l</sup>e � is decoded to generate an image �:

$$
x  \mathsf { v a e . d e c o d e } ( z )
$$

2. Forward Pass and Activation Capture: T<sup>h</sup>e enerated ima e � is assed t<sup>h</sup>rou <sup>h</sup> t<sup>h</sup>e neura<sup>l</sup> networ<sup>k</sup> net. We uti<sup>l</sup>ize a re istered hook function on t<sup>h</sup>e target <sup>l</sup>ayer to capture t<sup>h</sup>e activation o<sup>f</sup> t<sup>h</sup>e target unit during t<sup>h</sup>is <sup>f</sup>orward pass:

Th<sub>e</sub> h<sub>oo</sub>k f<sub>unc</sub>ti<sub>on re</sub>t<sub>r</sub>i<sub>eves</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>ac</sub>ti<sub>va</sub>ti<sub>on ac</sub>ti<sub>va</sub>ti<sub>on, w</sub>hi<sub>c</sub>h <sub>w</sub>ill b<sub>e use</sub>d i<sub>n</sub> th<sub>e</sub> l<sub>oss compu</sub>t<sub>a</sub>ti<sub>on.</sub> Thi<sub>s approac</sub>h <sub>a</sub>ll<sub>ows us</sub> t<sub>o</sub> <sub>mon</sub>it<sub>or</sub> <sub>an</sub>d <sub>man</sub>i<sub>pu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>ac</sub>ti<sub>va</sub>ti<sub>on</sub> <sub>o</sub>f <sub>spec</sub>ifi<sub>c</sub> <sub>un</sub>it<sub>s</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub>lt<sub>er</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>orwar</sub>d <sub>pass</sub> l<sub>og</sub>i<sub>c</sub> <sub>o</sub>f th<sub>e</sub> <sub>ne</sub>t<sub>wor</sub>k<sub>.</sub>

3. Compute Loss: We compute t<sup>h</sup>e <sup>l</sup>oss <sup>f</sup>unction, w<sup>h</sup>ic<sup>h</sup> consists o<sup>f</sup> two components: t<sup>h</sup>e activation term and t<sup>h</sup>e Ku<sup>ll</sup>bac<sup>k</sup>-Leib<sup>l</sup>er Diver<sub>g</sub>ence (KLD) re<sub>g</sub>ularization term.

\- Activation Term: Depending on t<sup>h</sup>e optimization direction (maximize or minimize), we use t<sup>h</sup>e negative or positive o<sup>f</sup> t<sup>h</sup>e target unit<sup>’</sup>s activation:

$$
\mathrm { l o s s = \left\{ ^ { - a c t i v a t i o n , } _ { \mathrm { i f \ t a r g e t = \mathrm { ^ { \mathrm { c } } m a x ^ { \prime } } } } \right.}  
$$

\- KLD Regularization Term: T<sup>h</sup>e KLD term ensures t<sup>h</sup>at t<sup>h</sup>e <sup>l</sup>atent variab<sup>l</sup>e � remains wit<sup>h</sup>in a p<sup>l</sup>ausib<sup>l</sup>e region o<sup>f</sup> t<sup>h</sup>e <sup>l</sup>atent space. We consider two met<sup>h</sup>od <sup>f</sup>or t<sup>h</sup>e KLD ca<sup>l</sup>cu<sup>l</sup>ation. One is our method, and t<sup>h</sup>e ot<sup>h</sup>er one is using t<sup>h</sup>e <sup>l</sup>atent space o<sup>f</sup> Standard VAE to compute KLD <sup>l</sup>oss, w<sup>h</sup>ic<sup>h</sup> is our baseline

• Our Method: We ca<sup>l</sup>cu<sup>l</sup>ate t<sup>h</sup>e KLD using a modi<sup>fi</sup>ed prior t<sup>h</sup>at incorporates t<sup>h</sup>e desired c<sup>h</sup>ange in activation. Speci<sup>fi</sup>ca<sup>ll</sup>y, t<sup>h</sup>e prior mean $\mu ^ { \mathrm { p r i o r } }$ is a<sup>d</sup>juste<sup>d b</sup>ase<sup>d</sup> on t<sup>h</sup>e target <sup>d</sup>irection:

$$
\mu ^ { \mathrm { p r i o r } } = { \left\{ \begin{array} { l l } { + 1 , } & { { \mathrm { i f ~ t a r g e t = \mathrm { ^ c m a x " } } } } \\ { - 1 , } & { { \mathrm { i f ~ t a r g e t = \mathrm { ^ c m i n " } } } } \end{array} \right. }
$$

Th<sub>e</sub> KLD i<sub>s</sub> th<sub>e</sub>n <sub>co</sub>m<sub>pu</sub>t<sub>e</sub>d <sub>as</sub>:

$$
\begin{array} { c } { { \displaystyle \mathrm { K L D } _ { \mathrm { o u r } } ( z ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d _ { z } } \left[ 2 \log \left( \frac { \sigma _ { i } ^ { \mathrm { p r i o r } } } { \sigma _ { i } } \right) - 1 \right. } } \\ { { \displaystyle \left. + \frac { \sigma _ { i } ^ { 2 } + ( \mu _ { i } - \mu _ { i } ^ { \mathrm { p r i o r } } ) ^ { 2 } } { ( \sigma _ { i } ^ { \mathrm { p r i o r } } ) ^ { 2 } } \right] } } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $\sigma _ { i } ^ { \mathrm { p r i o r } }$ is set to a small constant (e.<sub>g</sub>., 0.1).

• Standard VAE: We ca<sup>l</sup>cu<sup>l</sup>ate t<sup>h</sup>e KLD using t<sup>h</sup>e standard Gaussian prior wit<sup>h</sup> zero mean and unit variance:

$$
\mathrm { K L D } _ { \mathrm { s t a n d a r d } } ( z ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d _ { z } } \left( \mu _ { i } ^ { 2 } + \sigma _ { i } ^ { 2 } - \log ( \sigma _ { i } ^ { 2 } ) - 1 \right)
$$

\- Total Objective Function: T<sup>h</sup>e tota<sup>l</sup> <sup>l</sup>oss combines t<sup>h</sup>e activation <sup>l</sup>oss and t<sup>h</sup>e KLD regu<sup>l</sup>arization:

$$
\mathrm { l o s s } = \mathrm { a c t i v a t i o n \_ l o s s } + \omega _ { \mathrm { K L D } } \times \mathrm { K L D } ( z )
$$

where KLD(�) is either $\mathrm { K L D } _ { \mathrm { o u r } } ( z )$ or $\mathrm { K L D } _ { \mathrm { s t a n d a r d } } ( z )$ d<sub>epen</sub>di<sub>ng</sub> <sub>on</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e.</sub>

4. Backpropagation: We per<sup>f</sup>orm bac<sup>k</sup>propagation to compute t<sup>h</sup>e gradients o<sup>f</sup> t<sup>h</sup>e <sup>l</sup>oss wit<sup>h</sup> respect to � and update � using t<sup>h</sup>e optimizer.

$$
\mathrm { \ o p t i m i z e r . z e r o \_ g r a d ( ) }
$$

loss.backward()

## o<sub>p</sub>timizer.ste<sub>p</sub>()

Configurations: During t<sup>h</sup>e imp<sup>l</sup>ementation, <sup>l</sup>earning rate was set to 0.1, num\_iterations to 10000 and $\omega _ { \mathrm { K L D } }$ to 0.001.

## D Predictive Power of In-silico brain simulator

Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng p</sub>l<sub>o</sub>t<sub>s compare o</sub>b<sub>serve</sub>d <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>responses</sub> f<sub>or</sub> th<sub>e</sub> i<sub>n-s</sub>ili<sub>co</sub> b<sub>ra</sub>i<sub>n s</sub>i<sub>mu</sub>l<sub>a</sub>t<sub>or</sub> d<sub>escr</sub>ib<sub>e</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4<sub>.</sub>2<sub>.</sub> Th<sub>e s</sub>i<sub>mu</sub>l<sub>a</sub>t<sub>or uses a</sub> frozen DINO-ViT feature extractor and a sin<sub>g</sub>le-la<sub>y</sub>er ViT ensemble over its 12 la<sub>y</sub>er out<sub>p</sub>uts. Se<sub>p</sub>arate simulators are trained for diferent ROI <sub>g</sub>rou<sub>p</sub>s. In these fMRI <sub>p</sub>lots<sub>,</sub> the dis<sub>p</sub>la<sub>y</sub>ed correlation is the arithmetic mean of the Pearson correlations com<sub>p</sub>uted se<sub>p</sub>aratel<sub>y</sub> for the ROIs in each <sub>g</sub>ro<sub>up</sub>.

Actual vs Predicted fMRI Activations R Correlation: 0.7334  
![](images/038aa20b2ce9116aa4e3d8b25dfac7ccb49eb9068a97c64edf0765c2d80df3a6.jpg)  
Figure S23: Observed versus predicted responses. ROIs: OFA, FFA-1, EBA, VWFA-1, OPA, PPA, RSC, OWFA.

Actual vs Predicted fMRI Activations R Correlation: 0.4473  
![](images/21179ffa3b2f687d7fdcf17468c1bb10e62b7a6b827920d825fc1acf6757a9d4.jpg)  
Figure S24: Observed versus predicted responses. ROIs: IPS4, SPL1, TO1, PHC2, VO2, PGi, VVC, TE2a.

Actual vs Predicted fMRI Activations R Correlation: 0.3699  
![](images/505919f541086e3bc0929316cf5e5a9c7feea2b46dd9cd9327247c7492fe9aad.jpg)  
Figure S25: Observed versus predicted responses. ROIs: STSdp, 31a, 31pd, DVT, FST, Gp, PGs, PH.

Actual vs Predicted fMRI Activations R Correlation: 0.4360  
![](images/9df3ff5d1af11bf906094638da75eb104dda4be845400f13347fe222850b944c.jpg)  
Figure S26: Observed versus predicted responses. ROIs: TE2p, PHT, TPOJ1, TPOJ2, TPOJ3, IP0, IP1, IP2.

Actual vs Predicted fMRI Activations R Correlation: 0.5698  
![](images/4bcf7ea32ede94300dd12bd729498ea0240793f2118140d022044da1f75a311a.jpg)  
Figure S27: Observed versus predicted responses. ROIs: V1, V2, V3, V4, V8, LO1, LO2, V4t.

Actual vs Predicted EEG Activations R Correlation: 0.4635  
![](images/813e2d32b1cc052733741cf6295477b275a79a16e65540a9f7501e15244d4021.jpg)  
Figure S28: Observed versus predicted responses. Occipital Channels: O1, Oz, O2, PO7, PO3, POz, PO4, PO8.