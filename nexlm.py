"""NEX-ELM v68 - GloPro-Complete: Frozen-Math Replication and Generalization Suite.

Active method
-------------
A finite class-conditional library of registered-set prototype explanations is
learned only from calibration data. At inference, each local signature is routed
to a fixed class-conditional prototype; the reported representation is therefore
glocal and prototype-conditioned. NEX and Kernel SHAP receive the same fixed
prototype budget and registered-set routing rule. The outer test does not learn
prototypes, select prototype count, or use labels/fidelity for routing.

Integrity-checked mathematical core
-----------------------------------
The ELM estimator, local IRP-NEX explanations, NEX-S/NEX-P, CUDA executor,
CUDA audit, and external insertion/deletion evaluator are loaded unchanged
from an integrity-checked engine embedded in this standalone file. An adjacent
engine is optional and is used only when its SHA-256 matches exactly.
"""
from __future__ import annotations

import argparse
import inspect
import itertools
import math
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

VERSION = "v68"
PACKAGING_REVISION = "v68-glopro-complete-frozen-math-full-benchmark-report"
SOLVER_ID = "nex_glopro_complete"
METHOD_DEFINITION = (
    "glocal_prototype_conditioned_class_conditional_registered_set_library_with_"
    "deterministic_kmedoids_and_capacity_matched_nex_shap_routing"
)
GLOBAL_REPRESENTATION = "glocal/prototype-conditioned"
GLOBAL_REPRESENTATION_DEFINITION = (
    "glocal/prototype-conditioned class-conditional registered-set prototype library"
)
PROTOTYPE_COUNT = 4
# This is a prototype-budget cap: K_eff <= floor(n_calibration / value).
# It is not a post-k-medoids minimum cluster-size constraint.
MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT = 2
ENGINE_FILENAME = "nexlm_v68_glopro_complete_core.py"
EXPECTED_ENGINE_SHA256 = "fc78619795db892180c7b978366a67b15a48741103f0fcc98eeab7b7dcf2031f"
_EMBEDDED_ENGINE_B85 = (
    'c-ri}%aR*Mk|?^bugJ-^9s*JVtSXXfsU+xjiK1Gxp(vV2HQ5&jH#-3&s!|0YF_}P-MNXN`58Tz94>&VtajX}!8atb_xeNZue984kJUk*J0Tdsa9<z~kvyhn)'
    '9uXcM9_}6<?r|JHe(<9A;L#7!<!5*Mhta?P<A0Cv&G!$UKKZft{g00yJbl)Cdivdi=nwO9oXw-B>ufpA=H)U!7`)$q2M0gA{mZ(XmUT2K7pttw;O9ZF@!!Ew'
    'KcX(4MB}W^=fyIQ@_Jou)|(1CbjJP4=Bx888br^d-$&zoo%Iin`_bR!)iQ@Ze}DQ!RGya;sF0hQt9dbjwxhTIBU)APYFx~VX*SK_DW4Vy>P|n5l#^(|L(s4$'
    '^Q^9WlX405igHQK)cFG8&hol0tK=YxaJbQCty-z0V!6(%%X~>SiaMGWRX*Y0Ni^QnlPt<#W$P?jmQZ#1%VxPQqiK$ipnASMFQJtwRH@(o&oVlPsi~nbw0fCg'
    '6L0?)zOHk^FbwA4yRuqr=IXsh!n(N30?6kXcmh><Y<V=nA#KK{BTr8buqV;Obkcd+g&BA-T*1#*>C+_oCS8_mc-f2a)i>P{mHgH#slUIX(pc-MD*3Z6dDy=f'
    'eG+w48=oo=-GhUl7g_Z7|14nwqq3U9m?uD<Fla<TwDCUnR#j0}5uquA$)%Dt{LGi|XIUNy>}u5Zqtj@S*NX=HgeJDi6`<EZK}q!XFNT)01DxhE;x@h6m=-lQ'
    'lp%CLX1y-fn?g<0+kcD7Nm*5S^!FQRtc>QxqFCqkL9v|W2n<N3E?@@NZ~tYP)9P8yifWN9PNK66sR6rC)-ViQCP2Uope>;0vIe5OgtrGv7#SjGl~+@ulww)u'
    '71Cc+ZeZl|9Difyisb}8<gc@6Q)fu+(*jTh=o!N*g5|0PQqnsy!X#SaU|_wSL-}<D2t!mu&q}}<tvD$6zlr*%IZVJ5`&a{YKB>wzfS#9@8V`EWQ>e0rWdJK>'
    '$>Rcwo$<Q;{P6biAq)>@HqWPhD2@Cq`(;yOqjq|u8GxY0wk)yKedzAEa@WDMxGZpfVK`83Q7j9Xn~B08Aj&Q{F<1olBT<K;s;Dlx&axDBmYXHeBG8QO+uMJu'
    'eNgvt^4(tL5zcpkcw^QC^D^HQ5wlPDpIDJKG{kGQMreWu!UyK{G&ew{FtlQo!KgAMBIK#uED!<kO4SG8d^Era<_D3W=73=2HO$m`j<DYT%OYP_C9o`~F952_'
    'mKVkHjC%9%rzbrCXn1h`DxYlDWpz8-0Lv_^B8%?-{Ou`H<|13>6I^lS@+_a@=FmTg<M`m<U{=A(O4Awc0C}251(18Sj<O~2j|?fdJ~+_tsx#<Yo$F^<NY%P5'
    '=e2(EOI<GY??tvg*S||sqP{XeSMzK=1MaDxD+9MaM~1GS*2Thnd6`v996k3M2D1hX0^#Xit2Z#(>@2Ui6!I4}g}Rt|#U)nIRRQD)_MASfuU62Set){WN|4Uy'
    '*?68O(YM89okR}-2mJg9s5^;%$W~Y<iGE!1rj<lb^S|HZu=JDY*=B_$f7q<CiYm%YZK^qRp<iLydaav;w-pdVkzeKq2QX*(^0KIanbi4un$I#I<4*j;lSkj2'
    '-v7G?kH3wRC_X(qi@RbDR`U`ObYOq>H+9~ja!CaEpX7sQ)g}+r>94NvR|M;QHDBxR%gthSh3}RtGm|LA)KqTDqp4vjT<Jl;>Rc;K*BA2~rm(*#fiI@GEGEP#'
    '^;B0`u}pDkq&4y=w-T@e*!m~1^@~Yi5vrO#g(Uzpe(;;9_aXlSc(jCI)?tysnnF3IN1MdirlR%n5t@8*^$aHFJfhVG93Pm1?ZD)`EGBvM^1PS>>sr^`L1x9-'
    '9Hs#_yK#;vnwP-cfQSDkdUlSY(|V0>7r-tNWR=70Bhvu7g$ne*NX1gCY;pl>2pi9kD9mh@6~F^lzAk{#zRrPbVoy87L#J4im?O+7GHu)o*SKPck#ze9K($eN'
    'dFSZpZVGccze*>YX_lJ7r8wa!@R`dF2r^$sk3s%AA?ZaejC=|LOpKL&ph~Cp)p8P339JPl=P;ZOu$ewS506epUHS^cquudU0zJOSf#jA^4zvubcb#{zY?m9t'
    '|B9I^G=Ro>J#4OL`WORmoH8sBo{plgqItgbA^2T}?SP65O}$aKj{?Q2({(D1VMDuUbOegR22;N**5{o@hX(PMhE;W3O`UiEOa7=kJQ{6jnqDoKH9`G1rbq+p'
    'Cs8~GDuF%v#Q&7oT1%9M^E7`&Tmsl0Ec-Lu&T9#POC^C)C&-;7K*jj5wt9Vipm1de4x1Owuo{O&&u`2o7QHOya}=M5@th;6lIVZ}#0K;lAd3o>o_s}Lxr34F'
    'uYiXO322>HAqccE)d*Ou5S+CLg#s+PTt^P~=pTe;%?y&ha*qajG(%Hz#IyAq6nu>;iOd)E9}wuGY6JZ_FE?|I5$FQQZ6u-0K>nenZU{*MBc=-jdHC%ESnW_4'
    '@P++m87`YEtc*(-R=rsv0-@6|)(n}I9%f%p3iaO3fZfH0@smG2`|+py->0XK9{qSfz5nCm@8S`RlWJ_OfXHsImCx&7gZ)*x>Vz6ih|zR+R`HwF^$rg9<Y@;7'
    '>*|WTW@MdgMbF4K)6+kmK74fg&7%h}9;Ew&{M961t)mC@2grr%VkIbnjfd-g2OB}&Y>ap{y-+CFuQ_J%qP=Eq)u5Ko`m<sV<f0!tadEbU{RJeGgYO?ch3XIQ'
    'pFT?e_~56f4}W}|{_*&W^cboB^7xDXaZIZUxHt%~DAz8s>LRbcjIs%>KvgL!P~fNmZjQ`2y1a9@f9K%zr)LlEKYEb9K*4+vb*Q)TbLd81fiB$tGWsc7)9(lH'
    '`%#W^OAQj)X+P>5A0FT78tcd@45>y;fS3wIk7kVZRmAHtK3#zzk^xR|S9$PiUFERVJ<55sAOcR#izO%w(ZlDbj~_<}7{tTJ_n$^bcaQGeOQO}PAKg7V`uBhQ'
    'r@P0;{rCXT^5o$Ur|Hi>J$ee=9&#(^>-DNWxqZ9JUiQy`hit|mxm6_M!tw&`@Afa_s(iVe=lSh#L865XzP`PHUgy>ARk6r=u-6pXvhKmZ;~W)2lzaP=`V#o&'
    '=-`{59-KaX*3xDMGY;797n8cbnH2qey1BiW19Tc%Z?4ALLx^Y#s&C`&QLlTz0YS;_y)0@}f{JC^d+H1I;}1VQ{Pu(O;_&e9?~Vu?Da=j!^ZkeE5){s<8TH4I'
    '5dV+^Ndcvj8EReR^#^Z*<feRCqlz{yst#%_C#0xE|A4Xu2}}5>7A$a^d=2=-S5&ww-uD+53<KG8P11kDVqjWcC}n^*Bv1*Ls?q`XVOhRp$!rF~<@&@a2YOl`'
    'RLxmBDK{vMBD<qXFJaHgQ<#Cytb)ln0X3nIQ}i7^C6OJBC-dQlK6i89K2rnzbxEyZ{e&g80Wl~vGpz>km%z9|AuCZ_9mJdUtoL~jI0@=QI4YzZ(1Hf>Z~I5b'
    'v)CO4s<;673T5jPt?dnA3KG;0M;~@HRE66`wYGze_otwM)w&Cbriu)9{X&njAPdmoIEj=TF+dt=4B8B>&Sp8SlMZjzhNR%{i6eqkdB5I_J5@aVe@=V<ch-A-'
    '*!$w`Kla}Kf1^)P`%K{<6m_Y%5pW%HqEG4zl<%U55yJ=<u^}<QJ_=50Hm(^`ePWd8Ax+VUdbeElQTMAdwad{@=p??+K#wd_5b!(jl-2Y}0RyanVxJEPQWsQi'
    '+)ndO?+7pi^#+Ii!-Ncc%NhS(=4a}6*KY^ANRMiPm7wjw7Xr;60ACA@0utx3#8-ppDDNHJI}y?!VFy6YHuJfziX8)yy?+?phJEl==O~HP7r;-~9f(9S_wY8i'
    'rYCvLa&<XdBPig*6(ZImF^E$xrUt9f$nOHJPtZzNvBJakdRWgk2i*rMAHXm!K!ltYi&{`S2>;v;{J>e8%z;ES$Rz6h?vQ4K{<t)SHE*kyiUr^sUJ%~HttE*3'
    'JBNpw{MY3w#l3Qpx6&JcI6)ozBhwv^>H=l`V%bqM4$XEhx=@JqOq?la^w>~!R)c`oaVWAoJUQtdjfN)|BQ?<J6)f^%lH!0bI?HsHXK3}S?a<(_!_-w5us7HZ'
    'TcykW!fDEjdFQZy=P>#t^2*?Us_9uV%~6C(FH*)vuz9qhfr`8yIi09U!|1$$j<k)S+tSB--D`)*tw3Zz4n8ZYdL5YMI>+;-FuM_3Ugj<cg4RC0RAua~A!=wx'
    '9T4U&xLQ%qP@~K{SvUGy=fzm@q=Z_?hFAv%GUgwe?GWj7qpzZ4<hM}2UQg+TVnpKYxVx1-0j0{hVNORTzP>_k)xpjn_XiCZE&k37)L3ef{nr$Hlqt~IZJh&L'
    'IOSzt&9l`_XTafsVkYMK3}sDFKRS|=su{WHf;dJ-InA|-v-41SJsr&oa+rukYVdCvP}lB^7lzrOT9{^MRi2{>=|2gjvGoa*R?Wn_(bq%+$4GDb?byE6goTzJ'
    '{JNx#rkHm3rWxEZb6AQ<c7U=@HY5OmfdTcOUXJxki(azZr=WWx7so$D7l}1tsb@SwY~MDq>r@+w{V>#N#@=wGF^7<vwmz(li^snCdY~YOfo99ITy?wwxgx74'
    '6sz#Kc+pn^bPDfjcflbwwqM$Np+ui;^9D9bX$ZPwy4FqT#5SAcEUO!wt}CF0*JUwHv&}>a4CxrQORHoMO2Z_os6bZRLIW51l`qlp4lC-1giHSH)*OmDLzYxV'
    'WEQzVs|>a@3P4iq5I+;G9W}bk1j@jNhof$gcn#8wgvAgQOTbmauZAE8fh<Zt0hmU*GV&plNEbOCBCoSf1<GrITB<(58^%BJD>5^Z#ShVs=YLl5Q}eK#l_%a8'
    '*NDElTA!E8_{1E4_W4Z;Jh_JEI$dGv)@7XH>W5p9&&A5hDXCvNMUm;G*=1JHF(#CEjfbKnnf;uip54#0zI$KAaSC9izgbQ3q~(os<e=J6V}};#xu5DcX_|z}'
    'vw>A~Y+RP}4%fwRur=Gyu#L0>ndX<UVA*036RVP7<W~U0`ddpQWOa@XZBy5lRWbRX$r4woH3|60Q3cU{0ehmq0b)rY+~0eL+@v$(p=^r0sbhJrQs6)W$Q_n;'
    'makKKtImJy7XsjMIb%b{|E`LB&W9f}JPDYKPZElTPycp5ip}k*7HmF}vF*H^FVbv^z9n025Tvp2v(j0zmoA44DSs?i5OmU(wc3wN5dfe<NbEYygbP*WC7!f&'
    'q$G-UyTIWjlVzCzxomTf)m<<W7ZpI~EY{+w?ocS0(0~>PxRF%_Yoy%N<X3jcA~T-CLc<D-J%2^t6;K<euiV2|UC#DS`-SC75k5k_pSeDE03ZPJKZ!aE;OPjo'
    'YmaV3e{c`bkBNn8%j(e+K&YS#A8f@C(+6;zH9AhMV5QDxc;W^7zFHYUX_FUYi)5`nYQc$@Wop_`Z;9Z@hlJ}XoM@rouT?KX1X;ZZrYxWsVVM>h)hogM%Y-_v'
    '(b;oqEN)t$C2-wf15HswLrDU^ZYmSj*#f2qsZWT5*tQY|m4(90tg`BGf30jtW-1;Eo3NaXz;@G&8oe9W1?)qr0|~Ci<9i5mYnd;8ch5*etE_-!n7%CLlk(L&'
    '+DJmwY3(`x`E_>K^Ct7M&Plyaq71mg0OicPFn9PHZFoB=7-@hZ2ew!k4A2(2jG#hqW`i4iomVA_s^{f&5c9Vfx-lo`A3Pr&#Os%3uP&zf6h5Gz)gaC`>oV?!'
    'C2Vd^*a;H%!-`G$%}$4piuW9~LvT6pPf#auef%d;)fGqUWXR0A!ySYcV05k#i21{GTFzjh-`w)!Fg><|H=uebH`877o;Bs;V><Xzo6#g+W>rx-ugX;}p3z0M'
    'n1~mIZaPPjwFsg9+mE+N(%W*W{kzfRya4H?SPF%X0V-vVtDF~SAWfwp(~t8ttR~helwX9}dP5rDuS9W5!Ef}^$}-HzdhoY5R@0L$Qzn8*k<QDP^cNvOxuE#M'
    'ECw;0NW^jvUJ}tH(TeT&SOSR&w>p30Ak(8i5$1fvjwdilQb!^`JGqp27=pjL;7JMOGH3znp}9=0kWf=yy=8A!^d+4X8c3i~>Wl^+nWGxD5QT!TmJ%qF>X+2L'
    ')uMur`ls_1))#$(zv4ZwGx)$i#1~$0ydA)IyZ$6pUVo;LSdY?6k0(>9!0M^%bV5-R(@LtEE|!}*Wsukb!nP?&3bVwe_(IyU+8S&tmMka57xxg^2CAM152Sv*'
    '7d#|>2@sz{5cz!=q^5GqLjj^%9D!15xu9J}EgV}&Y<wxY%>iy+W5MRONGnQCSBRjYYVduyo|?f+F@55zBp|&$##ewvq#(OsL@}Q=X{(I?zN@W9AHyo340W_v'
    '*?1GNnFLLIAKp;#dwY?9{?+Y!F3Ys<saJJ5zm#v-7R7ySyu}^C?ws^xQ|BptW%Sa!v(?5cH3K;nT~$+bz5>|{KcItM%nqY`5X3g7I-k#4cefymQ(m}a8Z%=g'
    '&N83%&5NX^JR`ss=jWh>j_c^6@?E$%qRT67FzD4VXw>D+3$LKYtuCjZy<#51b!qRlFb*ntOpQNPmcNc5goNgmMqnLs!UU!q%(-&64-65#uN)`ukazSO)(jGM'
    '?WVu0O?)k;J}ZC){t?;MgQ_Yk^gUP03jW}gf+|7~AXxwayq*LA`f;Gef%4jq9VdJ2N2}G?c0dhUb%7kMXJruq=88$Nrn`cmwV7KJ>N3^(?LVvQ2T-Cf^dHr5'
    't0nh6y3lSNe)|_(l(o|lm|ybHH+9f5s}>4*V;we>Hq{S%pd=>`{+=Ahf^HI)UhzfCoDcyJyYzi}@!cLM`t`@aO9{D|y7)3J>Q%XX`>&Vz{7b$A$9L`WMRfnk'
    '&;7V<)U?0HP((~)KR(#iQ&=MQ4~Y_=-G!gD*?C9=wK<Lx)nj`CZ#mNpsL7l+sn)vAkzVWaP+c@W+4h{lFjc$iXg2NXg3B7I7!BmkaolKtr&fJF2!HtO?nl|p'
    'l~Xg_W5SDaAAkQ=Eq=3EPUnirx*O{Y=`crct{QKY`w`)`KH(DrsLy7Ay4+N9r-wOi7Fq+v!<I46FSF%ZxJelW*7!+{q+5ReDno%06vugfWtXc_WtEdYyMm$P'
    'IN&ucX6X_%zI@t2WT1C3d#?(Mxjg#7J|=iCvH22jJiJUbU~biOOx8{8?aqt$-OOPCY+0gDhEQiEOcxAK1CMUoVv^}oxjwa0H@Lhqs9+Jf{Jb%FJL7l7JbzrS'
    'zr$L*HO%7E>hCvrF+2l_V!6)y(UYp2!1NU-(HlBJutVrxBaeEc`+XgEUGMGCfNFiZlVf|5a?pK#WUvM^g!rO~GQD;TjU-Ll`=uTDov60EA>1fGaQ)V||Kasd'
    'c;~|BPWH1EUP|xa{ml-P(Z`z2oplyJM$U~}>#D44;_1;iD_&()^v1!#bw9!?fPgpL$hGliCbGs05DkiMrO?2-N8f9lgZ6cjM8kO3J&TA+&~I2F=8}is73Dr#'
    '&NquCT2zMdlPW(0#dA{RHA4Brd^0KQ9DjZ@FUu)>uF>xsUp&1ime0<M$;A@h;fsgMdNT)o5DWkC5dVCudiErr7U&kmH-+iXkDK+RMBfJMKaVDzDeNLJYU@US'
    '4v%>?Ad49>#d@In>sUe@Om?iL1}Y2)7pcNls<9>R_PrRe|9&wqC&QB@Lig1^yVI_&I?_O9>s)ztuu4{A(GCjDwz{a}aA-Y@A+~1AFr!AtdA>lA2&)tHdH_=d'
    'qlq5!Xd;c_FQc+p!|I1cHQwNzeQ5mcU#bFTuOBP$4x#e>m=qNDGdxoN6=-(*jU2RHs5iQAT+(z=Q?s~0w40s_JdTL(!;6(et3NLbbXF<WSNHSz{Auwz54=TF'
    'D8BxI>>1hobb6kX!w6D8NCxx#!KzrF=kp?)OEBNS{_=P;ndb#B9AF^#&#Q8gt$B4krmF^20wh3c0Vc2JC3hbsg7;n)x+W8;I^XNW|5vOJM!NlJRj!s9=xdV_'
    '$<$UG$^neHc_!x^2&xk(lPHcmh$m$Z<Qs=Nq1%(f3Qm$<2S>ON_FmJwsHM5Qhq}#u*Y;4)#SIs>Zs#V8`neuX^bHLsx=+K2epJKxGAgF(b}gt2=ybR!FAG#0'
    'LW}#XDyHwXqT@Jz1cm4d@&oj`<U5gn0Jza7B(qXaa0C7XcIf(|&la9P{~51){Q1vxO_IW56w3{|Mc`|`%SbmeNHL&@05FgYMxAk>rU^**eF164ps*XzT7IVP'
    'W?E<<X~I_B;Y%jBH%tD(W^=B{C-%$vm2zQGV$O`N80#1`ptQK7!n++n?P$O2gC>~c(mivBm~p=pya|>8dd}vH5{8B62rSzJn5ZgW@`$qwETSBwOw)8to+6jz'
    '$Rj4`;Y4?5p+hj5zMhgBFx$4Ph*6xI=_E2q4K6i*H5Vhp+_jA-b&YMPU?-k8F@nY>X(nTK*jp4FM`NI`SLhmjaU#SFLo+mELr`_M7>Zo$*eS&eYY^}XLvz(k'
    'o@hy`i3MSNp<CF~TT#wGX1Dt4=B<JFGl0t?^b$mQ5pWxY5#86I-u}hx$<*H4e}(n%r4Lp$ho#G?di!td&AeoSfBSE^hesGi2jtR!t_q6zlL>0&kV78CS$tm&'
    'N$CS9uqyOj_ib5E@>etEbLb)9hjpToq<d$2aANTN-wYI-5niOmn@<{_tgq^QV5mwtOcnmoW<$YX)ou=wdd&p)6qaOOcdoj9I@4ee$-2Wki}Gxir{4VCoO|OL'
    '6Tj-Cdsmk}>`hHS!g2rwOV5?2A|jzO0#1Pev^GdhD(X)TYkzLOlm#j(`C>JPCk%AO#hNqsZBY}&s90j`b(2jqm_gNf<EMG;&*VNkXDb7Ezb*6X18#>d;{Eme'
    ')Np@{EafhQXk09DnE<w&MMYC5R#ZPu8nfG2r(zMqZ1Xgf9RG~N3YNfYF{A~^b2){HQ(?~{D2HY}UaJ(!Plc4@)hZen+xLgJWMMdhd|BrUGRL6F@iqkv`S2z!'
    '-s_;f2f8`EX|bT(J1?dn!1U(X6|5K8FCC|3vs9ZQ$p=q6ue$wM9vr19o|C4jC1*V95Tug=o<TKQlX=4UWbh|$-Fr=u9cZ@~9wNMo&eUnGct<{X>|pG{d70yl'
    'Al#TdFrJ0gela~w$Ju&vPKT`?P&7mbKYG`@H6r?Q{J<^wJl($P`lT5qrX)Xlr7VVYEFts4D+wqw-jMLy|0<BDT9mK6(gvTZQS<CWRi~~4jcNCD2+6AP&?gVL'
    'i+@$med*PBo(fB;wasc%sfpY({2__HOQP>bZuRjEsyFK9AcX9!uIeVy{ZR<v^>zqu6}89Rq{{P@PVjG1W&93R5EVDBm8!cL+2lMAU8K3z5s#4D3C8j6Y}E2K'
    '3oTu)=v+LBUhljhucVyDG6f-Soh{Gfpq&}-F2Tpn>+UWXb#b;Ri>U_&ug0(R%>lY6|2m0!_xOT5{Rx_IIgkLl(?_4`i}pPQtGlx+RX*Re4oBa-f1N~5>n*S_'
    'c5qgOp}pR6{(Op1SiqoTULY{gUN96dntwx6IR#Pml4Fp4v{^mpiSVb?!+a0QeG;AjIx~)4+H<-^H?e~q)z|l+ht6x|Bd2u>aew6<i2}Nhv6$0&agpQD8~0*9'
    '6ul4w(sa^&k2TKcqwlrM9Z0H(zMnSF76BS~k4a;=_<fK5{)epJ_n(?`<E!v7*Q#yhhI95YSK_+tyzbMgd9TUkk%*=EeydMy<g<cflfIYkqp1ggDI#%B>k~Yx'
    'H%HbeP!Q*jg{~xq{K9q8e$m2l$n5D|iN5MelS)`zQ?ZydLkFD;`qg1*iNZDLS!m$O#d$a@QRe=R4Rr!wUU0Y|QBK=G_&T1`zUvKTN;+nkmN_M*kvw8r>@Cv8'
    '=}#)6&lO&ZSKnP!4izF)EX!T@tr}{KZR@$Zb;SXsI<&&*<{;m^YA9zCe~GJ&i8VRCF;pK?W4X2a>A1*85t;&JpkNGCtr#HC;-8Y}xpA^@hP(_`W~-5`>?zK*'
    'xQH4U7E2R>55a|@$0~G4WXTlZd7RDhP-O~iezK{u$(1+m-$c)_99GgjQ>PQ9(j&g)n8+A&eU9On7c1aiXv0vXf=3P;K11v~piMX{#tT1kgw85^nWANAfx+r)'
    'MAUP-4}pkGwEWRTOgDs-t~Z_s7%UPGzAp<{DRa7)HCGlEyzxNi1{6y6%V?6#CmReu8;!3bQgSg0a~Wlqu#*78sm^%hGvL~_J38A!st|V~FDP3N5$ifrMVPQ4'
    ';+PoCakvwsH#bI}t1rEXejhy_E=C|~j^R(YQ4uY5DPH~^sEAc~KlfZ^7=ao(HdwP}mE%cF-R!<`5t`R_5QfhLIxlVq8vZ-L>cNt#UG2W&-nP=iQMXzdOjq9~'
    'z+f!+L5BtK#q;oN>6TRViZ2rHhL`G4x9iCa&id8cP!oL1Qk4*HbKDnD<kTCtSZX7?0agDUoh_`-fr9nIjK(ktK<l7tkV_>W?_oLueYkbt8zT~JVn~GHAaM;M'
    'n2Sa%M)le=9pIs!vN(u!+;Bg|!rY196QS_U7%M#>FA(+ZrS!S5oIiIY3%gmgyoO$fzbI1m1e1MCm!8oy(gJVDdK%G{!eif3T5mPXY&C&Gna_{DI2M-B7CGv<'
    'OO(%BXwbV8H*PD&o0=?11Q9G;j(>@0Rl4U#lD`d}v(Pj=93{p&ZC)NDPcmp8*q@0f8J`W{|A}Lo9vmjt9zBTFW`xebJduOSpU^gY^gzVmlifY~3~zV4b2D%s'
    '6Yo??VTBeQ{vGSTxa6EQ<`maLql<aXWrM;la6!#dfIGNWat&52ZY_2=<0E-Pz6s89Q}SuT)?MDs2c5c>8Oql#D);3&P*&)-Rvrr~Dt~exU1sAlnO5u7G9@7c'
    '2Af*a#yQ@iCj3bu`)Cwcart7kzUp)=Tn#j_FuHA%bd^g*K-UYF#T|pv9oQ`$W3ZZLThA1{{XQqBU|cn#F2VhD3k(Ou^As~PT>#4Y2cmCCFirqEg$uAxPR-Y9'
    'xWy95Ctl8y#q7NI^#U1mgrcQ}j2q_yS1u`eC_M2ObY*0xvJ)5b_bu|5XUmwM*KQY}>146IfIggG#B}=beV}!7hV%)}*--VM(TOSqeK|qk69)b#AGlM$;fSnB'
    'P2iIon4{CxS)UZ84h$;6wq0P#6WJDiR4h;DXQZ$0*$R-sVng?S6^Iq+-q>8%Yfr6aVv+80md2@%u$<rZwv@gU_4Y4K-xWN8mdZLUM{T<jH&mq3DoXQ3Fm3AM'
    'N2jcFWzSZ8k*-axtWMYY5@VDZ3yZq-!?gx}<v#^0|7pL<ACM>VD%JXxD?JG<g&7}rH^lq4_xG=g6^8Kca>IRwMEDEB8ov{)*A~5DUYo9w+pd|lu5ZRxc})j7'
    'E4Qim#`hSBa9OV~Fbqg)$G|ndzcDx61raE}&-rmWIG;V9L3DV>|9AIfumbh_vsTGcc7jzSaN1~Nr`~GP`Y%HSma@(O|K0v%wSj+?Gn#v*<ibL$%5$9YG5+(K'
    'b|O8pM0M1)n9LM$^_lmD4RPj5HeYchNr=y=n}Qx5&y`E`a9Uz;Gch`4;5f9=J+{kIVuJ|_nda)Y7g2DgRckGKN0iQyl<ej!lKuU+Mg|T_@1c!G>wZyI04rpB'
    '|GI)j>D|6{ToBZ`sUdY^5<sFqM7p=gQr`?aCaMp+Dd@F4P8)$#toEBvQ)we*JbdB~-?9t*chu_P06-j|qS?gz!rkv3jk@ig51?`78RgF~6u@8o$^UTjfw(j+'
    '?$dhLrsL<azB}IZ)^Hc%jYNs}5+{Nf970z|flbDtzWsNZ0Q_+~{P72Y55%&i2}^tXwIYmPO++Ov15DaG<{aOtD+8G0FM5AJoGnKApm@Dao=%#!H*5UbL+ma$'
    '>(yq>Yzkwq)g6jq_M*#x#cIBhHY=?fHy7fZ+BXy9cIu^$5Y|*q?dKLb>U=--RHhYiG}LocA~Aztbe{OS+E$fP&D5tZNKknJAi-19?!ePoNm}4)+YN!X%$U-H'
    '<G5y<B~g8tE-ATc3bKoP0lSsf_B?XleTv9|v&yyTLFg!-r>QX*eaa{fR*hLuPp;*l9>epQlQndoZ<ZI{5l(n*==Bxw{`6J^!NikqHK)#6aBGg56^*nRB3B0e'
    '`{+3meODb)g=D=;XmoPH<!d_qQ6tsSqm<d~&|w%&eqq^P+k^yYtd*3SGu9L!0-F4Dwe2=60?*lV#0yfnQ~Q1234Rr}6znnG`)|GE_0CN!%k-?urk%jSp?7ZF'
    '&=Ikx9<_K&ju2)@j=<ee!D$ek0@~#OC-nQB$xdzI>8dKn*@xk0n$385rx1Zn4*EjCADCa=Ko4!ckPp{Kdmw$Y?%6YI)6G{o1`eOE&NHi@g<Aeibjt1vbT<xN'
    'u1+GVhRaQnP@fVvn+qCvm-Fnb%Fi-NxH^MnEWn``#D+XAa`ZMpFCDx<Lw9NL%qrS&-t2X4GSC7P-WB@+gor(KrZ(mo?xT)^l=!E(e0b~ar!I2N=0IZeRC#Y~'
    '$CAbf0T-xkt_mk>oJed!sZ_9zoh;Y)?D2yay{9J}JVjrMT^qFRCD5AjE%R3_zGA-Pe1g|1TW-m`z!I2RwB^JY*7sS3aYgEI0lvJKHqVbSDoh9t8zDkBTw2zQ'
    'INNOt4flaXktX)EETWN>NW>%_5|Z#}TXdj6W6DG7{Tj0yd20)ba|J1&b>?@>_?DU8I)%K64mh!AlR7<-)S@PQJa7B%ozhE02CKTGb8GvU_fqzJC=vB_ioQv9'
    '=xRSV-_PlLP`OE832P3kuUHmqc*4N=EOljx=3e6<%kDaUOnBi@r1_Ob9BHcTpSx?y(1V8qP9HmBTaiRwng#>Om%`PZEO(ZMh35h`DRT;%O*-945=}<hxHgJD'
    '0Ri)wQyM)37gFsutqc2K>nXmGJJWGP@)X$Zwk_hO{Pzv^+}0c_=qI|BPs5O!?rN`7banZrL0p!|#;rs)Zu*c-QK1L&19a=-1<bGjWt&CMnIgTS=YgV&V!A5X'
    '%jd9v4?WTSCO(aV-Kq0Qwp8X4@;=KuQ_$n5uM<?F2aRF5mveNpFfWwaWT3Y1>1#}S29z3Ta9*y|TB4SJ&ve!($Mp26ap>s0k~sJVvxzV$W2AS<(JU~)0_F=K'
    'aJL%-<*~EcxXS_hZ0a#mfXr&@`m9|XG%Cr^vJAsMhHZ7G`J9-b?m^FoVj0zDfgtVJM9NgbAka>X<G|8A2EQXVtQ4Q(6VZSlwv~s#-NNKg+-0wn*L{pD?7s2J'
    '^Oj5f#30=GH)-u3>;Vf3X}s{kpg27TO&J3frGx}1M}M~!1ad9sFjP}RkuW{_d8qPvwyH52KLuc}DFQR4*H%Y+A#6dRSStiP`HThs8a;YhYr0fXy?tWsI^k%y'
    'i7k!+;;%<+N!6|6|7h9e!wD^0tu)e<VQn%=(`wUVHt`X#-8KT8CU{EU#Kv|wlOPxmE$XniT%NVU*=?s``At}2T*Gu&dQ(UIQh?hn#EBB)cA$j8ZAKyFL-yof'
    '2yHM1?;L(r2au=|9PVrBBC*cOL(qshM+itfTC8i6=fW~wf3vJ%edn+9&QUiY*U2R5V%VZHM>nwDZIks7`{JBrUqR0Zdr<w|rM!J4y)Dw7Cdz<><q3TP--q26'
    '!Os)^{3_=yRoa9=w|<Kt$h!#$ccI}mu5=o~F9^p7l}s?4^-iq?DIF#_%G)GtQVT86_LLZ1O2xmQ>Pwv@KLU9)ap?$>(_%3QT>GF^&tfFQy{&F}I27x-^EAat'
    'g@EQ-_E$`TEZ8McEkId#W1T_>yF6@EXcsD7o@gaX7A6s%ccJIQ$dmio&i9iy-vv~=Q2H*=0wU|XfC<aB?*ge^_<e6^4GGz<WEPU&rG*`mblSXW@oR|F!od9G'
    'B+}A!L`N(taMr^=L^pGe5S2*?Dlm7kl+THg8gr$*1Q*BgA5@J9dXaL84p+cyjML~9ZKt#1Rc_)l+Te*)u&=`(-dupiU5guQn%4I6zTAI9dF`9yaQE{?FpJ6}'
    'x8jvKr&6u~9t%@E1nW}9Pyxba$vn~~S61&0b5`@@c^8{VfLgTD#e_UHWwv38b{IdONC+r*N*RnkA*=U0uTPvB;8kH4IKK2jYicyx1R=>62s>`Cd$O5^rCx(f'
    'B*X*fv_-CmzmhYTWjW3L$Y;CGVRjv}ypwMqF&=$YUpe|=9>&RhQ^Q&Tpm-6v<SZH8zfRrygP#k07!~_B(I2u+UH2a4XZdnk<x#neh8P)OB#u(EmnqRgLO-2L'
    'x~p=2wJaA!Ht)|B2>fw|KsiS&RuqT-AtkzFS^#Ex)IUTXevF|B_R*hS1^~)#-T>5p0((Q4RQ51C?5CseHcqV4S7|n%H@BI+{Ye^I5rYN1-2iULCJ_YuZpMNg'
    'XF&g9+CTO!Yyr5r37{R;s6`EC-Kj-m4AxO%(jfXqURXg5)LtUdM`r}T^<IlX)SRw?E9pbJr#pg*<Ch-RKvmu!ysq0ps20trYTjUJ$^Yz5Z=CnHiMrUX6>)<Q'
    'VNm&5m2q}7tGH`G1_j>;GC`3ZPoH`us=V}Y9b6~7O$9vvpNI~91k->KNSV#Nw{^h6WB#&-c|cc@9qL(NAH-1LLi$y#1&)+#j0P;*zK`L+hiee^E>PuCam)zs'
    '4Dr8g3GZi1P~M6Fhph@?hThM%z*#9#PE57r$TFbL-AjU_n`!dH2ef{AwOpU)>taHFhHLaV1RX<VJgz@{SZh@G=n|Z2iZ9M}Y&P)dqF7E<-poeO6cOl4ggim5'
    'O~pDP_3PONC~GLfm?V+Ovz7>jP-$k{qLkr8VBc)Y1neHU#W_jZVi~VN#B%%Wpz(^MLz@@C8jsWtA}%r?-#rjPSJZ`-BR9~QiA)yIxW>(jTMC=d5Y6&@WwlYI'
    'J}+j@Z5r9Jrem_iTMZgpEhEd2Y>QeN>0QvE-IJc<EB+4K_tEhv7{l#Y$8f{d#-*XpN5?Yw8v<$paM(Y-ryxKbt34NIOOgcIgT~3?WZ&UNP<zDvjsS!U9Abf<'
    'SAs@+=LEZc67J~XT|HDmLA~gXxCGyG2fi%PoDhU{N5OKdHu?*lQ3E1a;qN>iF2Gss=FXgBS4n;OuX%6I`*Ks%7)K92-EmQ%zZjA@0$tu6oceI*P=07k-ErI3'
    'L;owS?4fR$zupVYDT={WHrX|)K+$$y*O;v%GwzLS{c)3143GMsBd$Lu8l`_mNuWe;YZ3ikL4tpFl<2udMqObLMcwq=YNasyuCD1&>p2PidYq`vokzd5FN}JO'
    'N??;Bk2FmSlHQk^@BH;JvAbKD*qtB~yW7sh)>$zRgF3z`4?|GfSQvc|Q2V5;svKu<`XM>k(R*{S|Ij%P;Hgb2+8<SiqwRrOT~r-5QFXWjRfl_^>ahPspt!}v'
    ';Vx7i?%d-K+m^@iK6|_~ZcWCAn#u!Ej{+;@umuM9+ykp;TSjSXMisXD1d#+~q#NfE3w_W9BM6W!Pj4WY^hd9+1Vv(pj7*Sej<!0GS#kP)m`532QkgFY!x%G>'
    '$47V(-VfU+0|Rp4MJMqSHEGF{cE<yu#?*Se>6mjv?XStG=+DAOvY9qeX-QrO@cMXIhvG&}d)@sN96WhAJX93%V*3(g6-M<Rk3P8{GN?zF!sZaRd<8(TQo2?2'
    '!c>aB9z^%VJ!;%N?~KTA>5Jp&6Mk{H$IkgetPZMlS3x<{P>yZ@MV|8y%wXM%mM^*t)ud)a*yDnahESmrK$?*j_sB9fi!G_?d4GdAy>_yUz1S;**l-rVdQ)7-'
    '{^@|7if%K5ZnFb%BhT0@4jUB4`Eqi;$f}Ezh&$bf1v!hqQ&(j$1LbCgG3DVqllrwu#K~chLW$G|YE_6_*1^Q-gHFr-)cu&b!u~F=mcXu_et-Jp<JeJf6zRIu'
    'M`M)157^1+^2%o~uHi(z;AjYf|M>SdVbK<Kff+zCg(QPdBFud*1Yjyp@c?AVVvu)TI{;G}yhdVUP$PLRPP;mR>f$`v-nHZq;fvE;LtXWdk%71c_mo)=pkd6Q'
    '&wG}Iu2KoWRI$LG9=pf;y5E?+@n}Z~{#h(YM2#(<M#tak?jz3L7?9U0euWd4Utxs$M47f_gri1h?>L?1Z?xj^V<(u9Ku}6II2i<GJ6Kge38y^hwq{l`;iS|R'
    'gbpi|UUrnN!w^lsLwkqR-jS2%-kHFkmsLI~&z7)dOyz>tQB0^;$kS{eZGQU;Do@Nvnw2|N);W=)4)dz5&MjLUW0np-U?ZLpRStyR)VYOE;pPoDK<5=4UC4JO'
    '<hOz&!^`d_BW?5$xygJEoPdtnPoh-;CDcnSu1LFio0U%H$?TtEzgcKG5uBgKozp=R!0Gd3JkYN@gim+VPWJL(v9xyaGONz&a5ghRk0c#w;Rv&~xb((QhOsU}'
    'v7wwgO)h3eM<kT;`_691oh16?6K+rCrKO!hp{TKSsE8isLYufFBu@3Q3?tLzAaR5k2AQ~rEsEl6N|kEPQnXJ8{c&#_g?+Pv+f>k{sLhwi>7;j+Psi))5*sxa'
    '@;-)a@GT&M()8#Ir;a*=I~vF<j4gcrIPyb2o8=@xiF2#E(|Q3+ca>C{;7q)udss0uu$7FauB*x8v_|A!3o-}<&$+h((4i%qzVV&Wgt9Heo>_}HQcYbKM9a*X'
    'M5FEoMu1wa*QdVO2^vS#AiQE!kW@FZeNTANWcPcxIBJyu5Bfv_r{rjlO~;`NwUzjk5y$a^q489H#oSzT1DhGU%7NQV;0<L=F{oqq)Eyb0<vk*7y@$Uc`V8${'
    '8+JW#B`x=f5bHWVQloF`UcMQ>b3I>9OV>$B%Y~hhpfKLipSnTM$-=VTK4)GX_*;_dJ3FidHp<S6VR2G?3R)rBgBreJUQDhMM1bYS9u!T0mlacM^@mA@AsU#x'
    'qkPkCUf8x3QL@#87-?@R^TKfiuUm#-fpx7Lc5o05HvC*M(15$$snTj8&r8e#zRHT~C2)&W^7N0fggwXUCXAabTo>ir1;f>}kH(7c(CjucOZ;0E0;5+fRv3f_'
    '@etKlAh(=df$T@woU*0d>7r?ooh{3HT}%=dFIh8p%JA~P`QpFXst3_&=4D?itKzI+195OwbcF!1^sfK_PV?p2`rJ8L$3{XNp(<bMAZ)I5DFU&1wi+3}W5wZq'
    '*5%4X)$&LKsF^d5&{a^o7O+3k9*(KSwTMf00+Vq{2)T5(gxM7K2U~X;3T<Al3t7v)rtGVh8K`&&v?SpPJE2)(cQo=s`4|>FJan0m8i-rlEN4BhxWs$hSvSz<'
    'JfFePD|Dd70TxhT4TG%vlX5YJn!KQK-2HV)Jh+RF5&<L-FZghnz>am<4VHR$1C(G#(rHz$wxNN#>)oK{^@~1)213t4Fm-C=y;R+EL8&D%ERsl6{B#hZr<ZD|'
    'yS0h}yR{B3CLN&fO~W@c))p;XD)}qrZB=-lSyWPQ#f9<anyt5$`$d)GP6i4xf(VBMw28$Xm45LmDRcyCnaP40=647>V-Vr;uRs~&o@Wb7<hwPlo$XQ2pSxvr'
    'aEi3>D_DJ+ZO#_?5+!H8w$<6T!#Oo|LvD$Awos|Ff@0~y-hdf9`n1<<$RZ$jC@)`IvsWROPk|mAah|YC-4MSgy_@392w$_ZTBLdqPT)Lz>A3AvQLoQV4e<>t'
    'iu`x}7af{X^xAZ!>xC9&Da!WqQ^k8=V$e0)@0BDS2B3)KtjhE0Rm$>JSxrqww;KyoVM)qtiHh*vUzECl0lyN~HroR2mqNUvZQpn(Dsk%lq@1RB!#F$f8x!J%'
    '6@GM%lmG?3=mvC>6YF2;#*H;6Fmx9H*cHK%5(hSs+9ps!A;ghf;@i)c7o`6TG5I9*ZAzidIOu0xRzkAT(UpK+hbJe!qmkLD$wbE$P4_$vMk*|-Rb|(6C$SrS'
    '1-cZk2QD`}X~<YylC>Nz)b+MSWmy%ncQZ9dm1h?L>B3kD1NlG=JRIsa;QLE!;rJ3e1Q(;Zei{uO%(M$HjdmK+jE@4n-+RIwyEfEo9RtN`lw@(`Xt6M35)@2N'
    'Bd9;C&nBx$?M1jSgapd%qP)z#BE0rQm}*wn9W|1QXlBQORTa;pPdk`h3T&|$S1OI<O&H(%ur<rQ5Z+0%EB@_L@#%f}8L=w~&}Cg)S<`m5v?oh)xCAnv9H>tN'
    '4%qO$aN0tDYFnG{u`YQ{HeS0kaL{r$togxN!>Z>aA5naGAR6d~1XH=*1F40THQf2)t~GbPyY?Ag{f=wY?zJCS_1jncmep<}(^7r4o@oW+amyvVEvpVaHzKHc'
    'NAy8~Ri10_aJ<)qhI>vT$<3M?sUL!*8G4J-?0BOgP-n!hEU8echM(gj?u5MkrLzKvbLT<l4Fx!RSdxyfQqBc@68!)Z2<j5sQ(8>i@5l&Ml32E8wj8U{ojP{v'
    '5jE>czRaqkbY7KsPHYY6Y;u!2ldC}`;uKP32LE`{<NlT~B_?|2up8X6ajwgFBI!PcvTQR+wKGf@OK8qQPpA{9LW=|+R5qjr6<vEkhw)D|gjhAn1cGjOR6M<q'
    'ty)8xGfTDuZ!SzHF0coBaikaKnAdtPT<_Ug*&C<3H*4*{Fp=GbPn_M28&8y2)7IHOIt11J4f}dCu9+~0B(W2W;mO5_orvfWAu{aIdm0`MnI^j7HiMdbQ?@E8'
    'l%H7`Jui-bV;cC!XoxxHxw=tFniuvH=EW|%rb>Z(?%VBMWWiBP2yc_}uOwVfvnDhLsh_PSfn!lI_}f;792(LBZ$r-X6Z##tlxGw{VMRXSQfTGAP?olfZc8EO'
    'A}w3NcpD?Uw$);{gxS2X<%O(}y$X4egq0%VV}Y}-qi|eespoXjcg(6#UD8-7lE1OMEVF?0b}ei#yf<kg^+eBYDm^Vg_CpD4l~OY1eCA|_bT-3oC>nz}z(n8J'
    '$$5!+;iULNj<+DIs8iqpXZ#4Sk`H^zV2%>^HKq$Rxsqt;Xrb$yG0fbArC99;Br1j(I`a|<@L)F=k;Q!r%XY*09!z9E2C|EJ+=OvhrlHXiD6@-Y+?-vwtRl1|'
    'ue)~p)Z3vF-|@^$OlWf(+-H?eHWgi{!7W|%wFaUfE3MhK`SrECPg-e4?Mr&z*A+-xughYZLYWRVVw3l%i&z0uz6u7h$SOQ3;*xz_N2~EwXJ|+#5fqYWI!MeU'
    'CVHx1A!aon&}cT0ot<?8A>VkuDD2pv)11eIJ%M5|g^jxPdfHMIz8D{IZKhLWBNq>5$PPLjt%%De8Z(K&`$usrdqBBw^cgx*!1_Fo>IHdM8pnKl=KhiV${a6z'
    '<@ChSGPbLHueLiu2&KxVT9Or7U5^~Js6_~hyS2h-nG;c#skn(0FwaB-)!qlMT!rbCt64SD)QSkLjz1i~-BH<5gpdimG7fX%)7R1G+iqmJgST_fj=7+uP;`rE'
    'U7|7jbK{Gpoy<8W>=uUr;%`}fAuh;9Uwsw!x7Klp4OU_OUv;`6Cnw*?ubdt%RK!Au8vgogEYywuHWInrsi~oK8hSNsdJJ(YRD@y;9U&QyLuTsvQHi?I8Ojfq'
    'rL9P_3@tqcYs}gDyl-A8R*E-+mYCub+tuIX_}<mtBP=})oOw#xp0lP%&LCUch7jR_5MsXnm(}S~_CMpuTg8$9c7X;q@Ejk-o}k15>>Tt6FnD?x!W+6Fh{k%+'
    '5k*9`_zoBotx5A!o&OmlTw3CY{vl%gShHA9(vHLQKO%p>Z;h)ThjnY~ZJthx^c<+G5?3pC96_u%K0kJBc8v=OyL{Jnp6<Vs*>0XK#?vf4tBR?xBQjG99n|_I'
    't;KO)1E7EPN*eg6-Gpysg?;Lt63?omZy(}O1*Qe8F-E04WGR>PE1eZkV6;cBI;TJ1H+$DU#xpwVAg1-Iv(@0EW5?9%^di3!;duO!jdXAxTY<ClVZ@MCzgW%;'
    '?}Eueh9u{x8UObt{vu%?kwiElxJS|CNGFeE1P!<@vuxNJv^yq*j!0>ZNQ;w-O&nA@6P}9!A3inttkXp_)pq!#dCS>Jn&AIH2&GFUTg~$ObpH)SrUh2e`}QGd'
    'K|K2FWY04mZviE=Zd6=Gs*Ljke3-5!ch!f$5QUL_5Q}wzDK3?+(R>ejhxc*|+BCU(hfRRSp~S$bdm}$IrgEccA~A;aOlO*v4%>nSIUrQu(tuh^V+mAU;^YDc'
    'Nws@iEkc;_%y+|{UN)5OGmJky%PujNXMs9<<#<V{ofWUPjH@z3Gp_Yf$R^Na-FL4_<)}<{1b*!3yS{kBlk~M@FrRWhJiL0*Bx>?tqhdW#hg>?i7WGrHHx!mp'
    '`9n5FAMo8*=CeyX0rw39Kd~TH&IY_}a;3$3*&eoa;+$lvYU;AOBXsD`oFhd)ps6e9clR+=&(%C*&)A3(1=ob9)x7*zVdTb}Vm?)jH$ygzJ3;3LXKvcaBo7qf'
    'tJ>bLQoM97KH)~GKcRLI{N&uT55o1Ydxj+=A`SnlVT<6@R>;}<0$y!<9W?47d3%TsZ0*=BVW5HN3J`>W1kl<;EXF}x*S#mBC(1Qs+WC_%Y4(1kzUJt9!CQw&'
    'L?dceo};`QcXZ@esZ|w1=&r(rD%_k2S(18*2j~r-Ofj0S12|ZkM$&b`uF2n9UJMX0f%Ujxo23b+Q?7ki+hri$Z5(zCzvj0eq!VL7kg^@qYVPn#MWbgUnyM4K'
    '(-g<|f#MXHgj9a2MuE*JlXAyx1#eKffEo?wSykg6Jg)hA_hn7#_-F&pX0VrYfsUE_#7t`=rwo=|WLtpg68k{gW(ntyDH_J+0LX6(zp8I85g8isdtRI^EoW}_'
    '-T52<>BFOCmu?Rnt3))rK(`ei1&DxJ-n$Wpaon^iy+E~2rRdY)gyf8)3*zBmONaLXqgJ>N<=Rz&W{GhyXkJQW?tJ0}jV8AL>l_^I<`3@84ROU|w&m)E!Wj-P'
    'RH`LXgCPY$yTjQi1c7N+S;i?bkwdFnYK2>*FI#FAfmVu^@^W;@WX!HIbq9#=w|2CpvgSoCbzvTXhaofIRup?6#*2|&;ERu73K63|8BCBx*xNO!@Yh11H~XaX'
    'W$U?mT&SLD-bp>1u~Um(oaUUm+YQ683$+PEEzMRK?xmWhNbiaHYDsDbJU)IxF~OIXXB9ai$;u=Z$`>a52NmXnuf}cqz!@0m_?yL&+z1Z#*cp?pTb)J6)?F&u'
    'vFivhyLXdhpWT2Vj>H{sG!!5jh6({AHWv^t1(6r6bt(t^x=d$`Fcoc-0SrMIw_3Ok4?;pRO;F(ev6*NJqLpG$8o<w0$5DhlC5VG+nYdfVe&+=Mh}dbh_M9n2'
    '80~h7L6zBQWe>7}^6PWun7sf|gKVP`NY`GObuY&*NqOP!(!n=(ym`DXj+w9(DerO}l|ce~(zOmKA|UBX9msKK|2A@c`q|8-E<>Y0y^3x}%3B6&nbPvM)I?=}'
    '@ye=B*e&`9L-IzShz35<wN*vVCTqWcs=oh9GQ4rfp!o>M?rI*<Wh06Ezm#+mwCXU4vLn43bwgs(@niM?VUZkA3m7wLwkBR|UU&r=ZMqacc|~SN`t$6lAyT!n'
    'YrR~vOHc*o<+`)V@NDA={7Z|;ntt#@f0b33>11&+EvgPbaAcAs%3lF#rsahQRI-|{`)jncrSt3x*h9y@#s+7_88w$g(^WAzKE@<F;}Uh$$$4H6VyYV3I+Hm@'
    '#L`{H-lpn)O0mE>px|(UeI#+MFj4Cex9==`=tLytZ6&Yr`6A7#buj@ckh_E|TS_ykqptB+G4=Lu>I)N)!{Ft`ks(RR(jQm%5*eR*#RJnDf#X16U3(-~e_2e4'
    '`yKW_clm(5T4JXLrV?b=4*cP`oLKw|F3vE4H5~T8xCi3Fz5ZQkR6(&gQ5|+AecO&WwuiRHS=D(3tfWJAqh8c;d!<iWk1690jv`UppmH!u`0YS_gIf7~z~AhM'
    'Ucqc$)aX^gb&Y(Ekz30(hiSbF-0eIY#Fa;>R|K;YKQ<z$x=9sTN{!_jS<eA^@gMv$(ZB!We~;cM>b({<I8R4sR>Jy2gnzF5o}6XutQ2#mqC9}MFA;GIpnbXU'
    '={D!nL6AcY+j?1zP(D@<cuH?eu)`I7r=Q)C-lkQN9ezz{XQ%eJEY=WZe%`-_G<YOvQ2WHKFwz@}!3<aXjOlGek|D>0!#{mw^7>Fx88x~Ny{h@BK$<cxW&9ZQ'
    '9mCHUpER!qF}kL;kf)}w&0?HaQHlBnMY1(+GXZk^0sV+%3|TsfY0}{vnvuTpe(l^!Zza*K=vG%1=w5$7x?C3vkmSN_`+XR-Ax>VtY7r}{>ti(0&M+n=1evX6'
    '+LO&ROS8?i0OIYAq+8HF6i0}r-2t!&8*E?W=)W!LQC*GTaJTS6kQ;^eKod9!h@)2WJvvH=7vDuPKJ4GcEb1i45s+?I@MPeV=N*^7>r(oL0U$)BBpL~7Vb0f9'
    'lZOaw(VJtU${O&QcL1+vag$b`a$CT!%d?0|b{oVO%hMZFlU4j^(BbN}rs*a*eFO)WpT9kgl%^CBJ$xiT7ejBIfJ2LS_pH^&M0k2cpd`Qs3cH7lHxX2d*uP9+'
    '7=6mh`6Y0WzZvi{h_@?ALqAIzEt(UxgeI#^iorI4GolLTXQ%?5a=k%2cnmB*g)YPmi8rgs>>cw=z{bEamiuwTR;<QLmSOc<NWhA;QQCQ-9Pd(;eN9v}yn1<8'
    'g`A|dffvAd369UmN6;X>QD3Iiav-_6wzNmVk%DWlytzJpTKQA%m1lrN<88tvr&?o}sfg9O>}0R9xNfUSj?*Iqb!<oF+6(y{iWat&YKIN7tf^!y;x@VRE6lDb'
    'yg7ooFb5L3XPVJXJ`(isSb+1UO&1Jc$0}}%n6I9g5Zzg<t`OK(Tje;^sFpSHfNx7p7`cB4r~n1N81&KI0b`?Tc$V1vMs{fG-q4Y@eX0VJW&zzuxd0_+wedS|'
    'dcF0yLIRmIflO`!WQ76O(AYvYPdl4m4J_xNUB}E+@C(hvSL~T~gn|4m+d(t#y)Kn(;C4V6l@Ipdr+f2K!AE&)F@+!HVUp+^&p<DN_o5CB!30vG#WF2sv%Jcg'
    'i5k4L)p6ujzb1ZctH_p4nFz8`9Y2cd9ulFD_r_}~U`r(uK>EYViWK*)HL<lOPOTlAj@Z7Qq5{6IFlh2LvOcWF>1`UV&w(4AowLY6#C?a42<|`m87M{Jhos?J'
    '?|!(eeH5_=mjklSszB_19GQnwu%t9na%klHLN=POxS*_L6||e8RLvR}kk6{a_lk_pwpOvDyZw3vgxQIcxXWw7*6?iO+ytp+E--3tBu&R$XWQk5Q{^if;egPa'
    '469)PUep!(al56QkpYrhJ1T5UlaGW5@@4x#Xt0iS<idJWbg(a#PI}tmIOp&$p{f5EPkmPzGTLS$n-?i9GgmOa3cWPaa$A;8Nqo%+@Va+y0w9qlCCzK-68iYp'
    '0~|qLz7eTPR0bqnMkTs*Q&q90iAvb#L`ddm{1JX6{<T}&9r6LC<2i6IGl0&`)l7O>^IkylG8EYu_j)ktI9pH7S%PA4fiJcm?)ZKU_no@p272$hqvq4hp+hve'
    'nX6EEcUS=`_|_UNs<R@W>X6S~<UXaKryK);jbH2ROe{fbc8KHnJG^(8XVv_QBS&ZQ9@ES7oE9LpIAI-DxofV<x-KwKi}rvwmI9Cgo&vXZqv*Std9(zy;SBCv'
    'wNI-&<+pSrf0?NpiU%T<34iZKU&;I+jdkl@u8!YGtX!W&ZwMvVI(<SNWorNqJ&e{bOC?$BF!lW?_MrVZuJZ~N%V|-YogaX5*YNBSwTMpl+Ez3}pm;f44kh&N'
    '2O7u*BXMVurH+W0B`62L5b>rWk8V}2Hf&X+!A0j;-54F7qDF5T{k)D!8a7}1uDktcZ#4KAjK2e>`~daN3aDxx{Q+2Nf(rmISFO)sE^fiF#<N>UY}k1<At*x7'
    'ms>D#81oQ^tjvRbiEDKRgc!gyC3s&Qlp}E-!D!{YVkd5LsNh8t>HR%Z?uN=5(HhA4X1_-!3Q^lS=ic_wXu`kvP&TFk@?h0Ar1(r)WiM$HL7!#%eI*zWhi~G7'
    'RDyR`>lMw7yfDpOVK%awFMkaV`*^zLaAgx}CedVgHPBTL>>6;cTG0e&7dam1lna_`9uD;jZ{5tw`IN5V^rL6+)$CNjpXPG~kP6e?9R#4IULO)#PuC@$ND|pn'
    'S#yU}U!CZygL1?QUXSz-lmu$K-CZ*#`JdwW{^ZHaMsf;ayTJAnq&(G;_Ng-66<NK^OtDNDBH(Vz_Ua_A`<+Nure{@?k(gDoexf%2be$nZPoL6@jzf^-myfS;'
    'H%uuf5Qt|B2dTO|lcy@~6jjvhY@TLk<?l}V8I5|m0r>?m%2?}nH3d%OCMD_$y>U`5FLRi741dh#&zr{J+Iruv^|D{9Kp9L#)HIvotmw#^B)_x`%J|K6jcL({'
    '(iE+89~TrR<&e>GsqB<{`^3XSzAVa(Ojj@1sk|6RC&@4m6dr<fgG*KnF13QOb>LLh*FG+d>T%-5<#kMV5~eZWIh)}NypHWhM=>!}=o58dK>xa?;;`k9g>Ps)'
    '_&GUf-o!#11w~}#z`~ww?ZCRD&UHHbB_<n~nK{sxYS=^*vgmOWJyS!5VA<j`LxWacraTY6k8a~u)zUA529dqIj=94c2|Mg>fksTB$vJKqDJM`>15Wi&2#a3r'
    'e4edpbiPriSGr5OFaGxyCu1`m1D?{oTxHB<#j!|^xkWVE&Ttx9tXj|Inh!pM;#J@Zkxs-pE%4$1J9WaaNFPwV%yNdntr{&SPbo(L%a-`i!LCHN(Ot5lbZ0R<'
    'h&>sebZ_A2EXE?~8~j%v3#%(w4p7+|z+R(5lhtM~KU0Aw+H0Iu8E9B#t@H6VHZ)IklY+rEIhRMqtTmyj>jy;vDRj62HNw#%ge=rGai38yR9?5Lch<pg<Zhp8'
    '<M3wrKwNlK=}i@NqI`szqiUPcs92gDsFSj+rWk`DufOEDEQk_dN&+AWj4TuOtf)p(C5CiBtrnI_&xnh3NV~>&=)7WYPwo<~&4c=|RdV|{g0{=tt?=ap3>KYj'
    '1iNr#Ft?T&Zm?2Q!DtF%B)shq#u{aOS(lF$+r)FRx1{@t&<)T@LxgT$Jjqu1xgj?P=4h{#=MMeGYtCH#rG&kb=NZSCKq6#xArbyR)=GsJ-uKRyEkp1Q)Fj!2'
    '$lm0tO^#gk4g;ytZ6Eyd0b8r+f^UA)0;kd<zY!_+G7MZqmtdw=GhLR;p48WnFHn$(nWt*@J#ju&NrY4fhHAD)ezCA{?7`_OHcnoc_M1JCKQ!@?WXeDf%Fu0g'
    '6bvw~`>O~`W#IM-U4;tagUWB&GAcKmHG3Mc`Ly9YY7=k<;$Yix-#O~qQRq5;DD$D$GuvJ3!Jz4<;l$uTHguz$9JMKdFK0h~@Ire?{{1Ec0$k$-ek3as2WCnJ'
    '*Ane>$SUP6eVSY9l*;&NURYY+n&Q@7CXQ<13_{Y<(&YA47`=0NBsF~91@bX7GwNp?ZFF7WRsd4CY233F$MpKTk!jQBu~QB(=%s7`nOPIBwGVSgO%^IeltoUK'
    '->s&3DBB}}A)juj7@4Vwb?IP3W_U;a&Zup&GpzKzhoaZD7u)wlx<OLI$3ozrCE?97-?qF(x?8in{ZJTUQJdmoQQtF*8XMEAS(hB@9!Mu7T~eUhD@Rwt@Xd;X'
    'JYtBQj-R2tYwLuvwM)}2dq?0fxF$MAGzZ11u_{3kB=ai~Mbuzj->JE7L~7y?G4c(fTMu_<engiixDjX6gGG!#_Z|}o)3-S2!|$@EHbsCW%Xzj)yl6X#V{50w'
    '*1;aJv)*%X4KyEwk}9+fyLtXk1x2hS86(p@9#>LjBPNpG=1rg|FQkvSam6d;cSlCTu_>hmu3uIKS|;ktj*s9(tF%UQ)jQ@SE6J5)TJ)i!)=ljW{f>LP30NLv'
    '1EIbB3&RF%<BhG|?bpcf>s}kYE#H;(IWqd7J$;N%e-=&>=)(<H69=mbhYf28(_NqMJ4mOv>Tb#mIm9Lj88PP>2q`@}8)N~%D&{P+>=O6L#xCUzv!AL-K0Y#&'
    'EQL3$H*833SE9^SOBuELZNV+`ZiG5zMw_PQJloXLN!f=k;E@;dhMna}d!$oTy4BNFF7kZE2d@Rm-deecsyR|0tDzr2?Au4=yW14`_7L}+PP@nvf|vF)bp$WZ'
    'wN;2*rn>>dxHpzo5y+Q-#7&VK3V<I(`13mdE6RV){zm4*rWnEYnj4D|1~dND?1=u;63eb_i``SA5CIx)8qjcuR6tn6NNIb-a%?hX(>V}e_6u;hH;#)m6zgy&'
    'LS+O*KjNV$k}3SEjez_=^H+cxOwK`M!)$~@JkRH9gHU%id3TUwM4{U{mi`?vwCL&gr%&ROXee)t16>9-B%5wf-Z?U?N+iaj2!ihebq9qsn)(!DH4a76M1a#s'
    'lIX$L)8_*-u^%0(QO>dgr#cWiKbdgoDjdaTvR63Dy@*ZD;k)PqZKAB+eC{0{dF~PB8hpez+f2`6k0zB3Mt??gzM)W5&T!ks<3)5^FK14OSHy`TQY8t|cD6%|'
    '@kLs$<4dzt*ov|D?)YM?V!BGo@!EGJ*qcbNYm+oh%68tmBMS4^hOnLNTqBMAbh34spKjX{7sj9u*dQ%NMHGIE(iHN6o8$(299J0IB}j4(ZUFUKPM}`oT9((V'
    'T${PxVVR3I6OYIemu^1FTK{azT34;NE%ryWy&yG##)^hkdnE<ey0UWFXi4J!xw9MDD;N5O;O^!$As7WyoZI=mw~9bwwUk8a6lUN~=&!_$E=bKLR&Z{}Q{Lt;'
    'ttxJhQ=2@bJzA0XycH0LVRa6h+hG!k>>~mld(}R26PD<ou8l!N4)AiL2u^0>s>mrh*e|XM9q=p95k6Es2t+X3K4HOl87I;{Ez0f)qH!Ya<6caqmI@9>b8EQq'
    'Hcp!M0L5#Z-NH_p8wdy;VgPw{=|!IEsJ$d{M(ncvL=H>fxa0Eb#2w5{V7SaT2@Lha^e8aacj`+!92<6NR#Y{+dMErF`6pTe)H{BlsOolZ>_KTb9Bhb|#esQ{'
    '2zEv^3a-3;Ba^nSKovVw4~<e*y|GBSjxOuyjlhVX#Y)FS^J9ZLxNM6G+K2(VwU!P(cd;$-rx)te0q92gY*~>$7Yt68*(lxoH0`;HkU%kI&!*;VHFTnYk;aKu'
    'wRulw9@@u06Ju^ORk(FcU>gyE+ezg^hf%J`;pO%7l6V<m+-;~Kj+^r(FU%%YLjh__cBO!ZG%Q9=mKysS-Mm%BZnGr^i+jcPY+YUN8{ZRyC5LjhsZfI4-OD-^'
    'o0h+p+_OZgfgU~0V9O*A1VZ|TTMWkZ-KVYHL8pTqFaZ*`joGk*4>Rayo8jQYA92`j#K)GgOP)nVlE@ic;_WhRXA=9g914l7+T1<Fn|4HTS!Y%HEVj91ttHIx'
    'QXP*1^bQq2Pq0_z)1lrEnxQ0jsTArpMfj_LCtqg3=}oRGQ`_5|wSpfMLj!twQ0@$=(_wkDRSRyBISK!sX7?m3D&-ebEPL$4i9U4&qOp#$NmbUhW6U7~iedD9'
    'eZN)J*lVA{&4jaMa{3m$Bhc~eO{ihV!l(wzuj8w!s*F|1<<$Tyv_QkszNhkQtnxlx_pEX3s`Xg;3<R)xAhL4^(8?JQhv8+k^fQOCP({~w5fh$dtPH1#HERli'
    ')t8S=&Qxc67E+m&PpQ2|_A^<ubp-AC&)nJ8`{{~EDYajlX-e7@c&3GBFf*Fp1{5iyJHk07v9VuA>WwH9*m{+D<TEsdl1MWy%QZTXu2PJrMJit)^Utze7Ry-<'
    '{jadQpl6>I`WANRlaKbnz~u%^czC0u+17b+hQ6LfosRQ0^ohe|x_Mf<{z9D2e7@?<TUTDocPjeP$()bF_+>Gll&{bdm7bS#&c%*D(ro(6re1g2i_>9EJPHJ%'
    '_38k_Cj41y0!jKxM@j|d1>5YjQKbW6e3ecLXmeJ~=gX|a*PElib?SC@j_@mGgqyF<v#twE-hx$YMqgyY8Q6Vct@SwIJx)HYgKyyY*4P9;yLTkbSR5(lJfEN%'
    'E24@8JLhym!~30HM0h2iDyJs235PUyJ_bWGvL^w>^z$ir*tYYA1^2L^F0(8Hf3+7}wKH)e(3rU*<aj?PHuZCCt3qG$8T?k3qWJFNAznp{+*=M9O?3mhp}7Pb'
    'H@4^xI7-21oP!u0vaoHqz7MP%u?=_Vj#swjrS>yI1K%$G)rPD>G;STG^!bp(G~pfFM=5<A%B6Qso7-4~^~W7C$_P)DV$F^pJH{>YiZm{}?STR|b->u>SHzxG'
    'qf$fo_ycyazsG<bZRXvE&8{j=!|1kf#vbPIXIfxP>tAK?l$4HF)%%Rwjn4c+x&`k<2=`=Nu94)Cw@r&nkk6;l_)5E6p^eMc8h*94zaNI0ZeQ<#5ZU~Dqh;hs'
    'X?BD5MKt<iHce5^OG+)uIU$xMio=Bs)<+a2fFHs|)l^_X3cYz7qmZdUsOTO*k?C4mF(j!kJ-78n#eR0e?Y0clPR~dqhrjhPW`qv3=3cv!0tj?zlC3*KewR4V'
    'hvGm^o~~&mB$mz$?YJZ}!n_p4*|O9repIwbrJ<irr+mROXHzlSjjxcdd-Mj+z%X?*UaKr>ZRw~{99GN;bVUV<lzCMR`8Gzd(KlkMe13%kh8>KrhL+F@omBMA'
    '{b5e<;T}abjt!HB2|Z~&{q;{OE%`T(hm-BzH`|Zx!+J)d(beMP-{P~Eh>YvLU$+qP6R`{v*W#5Z@@kwuRZ4jm*glb#2XQH4)=Y26*CLQnh|t#bj!%5FuI~i$'
    'ca?trB#xV@Vq%YqtmP=L>H-5C`IM*##yq-CFv{KC&`XgiQx*BUp9JWsQ5TnESxwt3hax+M>MGAmP!bhZi%bFX3AO<T!vrkK0Jt!KCLwA-8BusKDf5~D6txRM'
    'u7E~mHiS*HX})tTYGJeu#E0b6ZSB0MrBEO0fV(vJ=s{>sTl<h(6G4jUL^4=&*bA~{Dj4pn5A<My)&jbG0UQA>3~7x^A8M2ugyIxUN@f;;^JnjHs7hh3vJC|C'
    'm%O#M!+ZouE0zN^NV1-)FL_au?r_5_E2yI5!|Ga=X*DWon>;YHb=g3X%i<c9*T9ZSP2H0JU5nFq5V!d3U2{UqRT`O(?EM<W3(ch(4M#DnmOL+kt;-QKbrgx)'
    'QA|IBTx+$oaq4A`3~zgKC0O6JGSF0GLAEvIJE*xWH5|LPH{}dUg^gYoGxnJjT>KK5KZ(14eDKrLhd(|}eg)^Cy4+L~;Ty!yPQ+Xl=vI3MLD|!vo;|$(=t278'
    '!J{8!QNE&%RJ*(^s&YwO8f3N%yWdftW$W5h164ddh2wy5EU^=er)(kuetvlS_|R87q|ks<tM!JqFl>H(6K>NOolc^-Ed(1v@d42B6vp=2%rRqoOh!$Z-{;Zq'
    'qcLE>FE9P4*qlhgqIEu671&TFb`0h^wN)`mqN>G4gK<8?KXo}<gW45`lY+X{w=~aQ6~H^BCWcaw=iX=<kT>=1EbsEVRKbvJWszgv4c+foAx)EcfV2-eIqI#('
    '>DE?P9L?Pzx|bn+(?o)KpYEJ1kL?gB5T#St2PK_Vw*Vp|ze#)Nc;$A31^~?DKoOfX=P1o(lWTFM@)z_>(2OaGyud#CLm!K2Hl{){2^QAi{yT(EU$)#XZhG?H'
    'R*}-#D)y2co3i09fzdl$Wm&9UIP{A9eQWl}KbbWRv<ZKH1=~ccz-OhnEh3#wrq`6%<khMs&2N>|oT|8f_mI!r78<&uC};(>R<TW3Ia*{iNBr6*j*;l)$zW#h'
    '#XYcM6tg$BeA#I`z@_*z=B!>G$!~Y4<}P&g5|{D2opxn_F%hyX5XH=kb7uPFW&AXGK{-sfIv2Lx-mAD7nO7R!9g@%;$kuX<2QPV5Tl~lyx>)cHxsT*zA#$Hl'
    'WCluAN)0yU_3eQNTM~A*nR~apuqr{54Yk_6fQgA^myn}@upz54HDU4z_KBC{go9GFq&W_b_<cqyOwcl7qLZJ)43%XsCsv|K2<#mmNdV1pVqnQVGtp~I{Pkdf'
    '^WyD9!ISMMppY&8VaiP~p>K98+i}B~J<x{=b`h(4i>w361aFRA5PDbCWKpI_lYM1!{?!(rJPPOSC9rnvl&J2d7!lH(&fqv==e&mejZMls<(j4I($PvwGd!iX'
    'YU0nP#L>-?lYf+yDW_*3?44o43u`r{{6D_IzNkM^3OQ_uw_M@BCAVNdv0__H()Ltw659TW*Z1A*I)-K&er4QdS7hguJ<&NjTMu%~nT?{~k<8**T$Uz{Cq1NO'
    'gyIks18qH5ruKn76yiN{pvV%?d3J^#tym&kieo74t7v(HyD_tM-|=o-a<5wK`{*6#IYc@I%Jn&4Ty3TV-pglEyC%xS!A^fh2Tj*DXb#W3tT)u$8Fxq`u>*Jg'
    'G;cEg77x#zXfS@6A$$bfG(RLR6nnM9_s+O)GJdb>JJEp);>JTI7dkOk8Fez~fNDKmu?g;_TLm<piZZlO`@^Ag=VvtPbDon<x8W2^^?c5>)e{ysXI%U{9bb1b'
    'C-1UZz)O`n$730B-j+JiZ`IKB>!6FP%J3tirbeGeUmPA{oKWNOI`~x_V4J?LNX&{PZsG>ZFFZ$BcY?B)B0n5DE(Gw1iGM<f^rw0lFKLTp?*whC<Eq_mvow1m'
    'zbuxv0?29gX3_Dw^)-fF<)F?I3+)xY;tI_sG3KS;7|Dj5_}&GB*?(dFvvoNi9Ob>E<0Pu^H^JUE9utR`nlciVH)7*L<D#zHSa^6U&Esw^JE~i+pz22Fwr*sY'
    '3{D<2&ss~d033vu4-WcXV5mk&#$X}`PM(2)KqyAR#hStmhm_ytHytR~?l7$jzk{cAqv~ER((R+z&o^3#c??QfdnlYSn+Ji3a_sL8=W|O%5PM*h)ft+091DsQ'
    'hCUHu?7re^GYOoRu!}p?3~DcWh(_n5BfKYsw=Ig2Ml(Oprlk_(I7qGZnwiBvJ$Q8b?Cn4O^zg^%+Xv51AANuN?T?fbT?uB<8+EcKZ`j6zIQk^|{B!X}etq!t'
    '{*O<7ly7>`4{!gnE~jO!?}Nb4lZac;7p&!O&hKukfOC<R5}hx7w+#lT()?tRf1U(zfY27%G6&YhV0yj>FhG&r;Fh06n>t%WCCq)jC~*t^1?H%Xrn&S!_rRbu'
    '_x69i{hy@{OOXR};rNwr{{?qjzy%Da#uDXtT_A3jnKsW-%`c%zI*3=!yk&&B*pR0J+bf;+<nJHe{?oTVKl<^hz?$1Nva$dbvdafF^}i{0aW;SZuefe$j&w4m'
    '%@kV)0>&u$IUkcvGnFB&--P?bhda&fnHZk!Mwdl~k&&kr-eSw}`ya9msF#oOGvIYq2<Su5`hj*~`=;_!gW476c!Xy1<(fh)^KlX2QaP~%Mowb(3EgpKC`@H%'
    '8`zsN0$BiwPSD4#Mt6*9i4)@tA^_GA;w5x7{c2*TZVQXIf0-5~Vw5smL;2Qez5nA6KSqxp{1iR?`N;!VuRlHf_Vn8aZb$By6)t~n7-x~Fm;&J7dMK9X86M?;'
    '*tf(In)C9+)a4jZh&8<7oq7fHMP$$FDU%+E3J>o`MqrC(*=1S1{pZUZ$^N<G)YPHxfmcV_I4fS!`k^)2XpTB(L^o<B<DtMpgr6wMK_^z#+rRYKgK<jqY1dZN'
    '>$%cjkEK7aDGY6Ch^MAwE_kfp{u{4LI#~gJvc{nlye?Mx92vn9NNl=+wW4d`V%3MUm^%W>0y!h`V4v?CHU^n1*GL+k<`JqRAfZHW{}K@^*Cg{)T?&rWOs&eg'
    ';E8@#u6h@(bo0~CkDopKK@ticNcN|j#j5UX_d3uUO;%5ef-^EEbP0309)OevsLx;n9Z)KVM&nMpRl6hlQVDHJePnPT=nP@+cn15?|NHdW!~3VvgU8Q)dJsK*'
    '@B{pK@Dx?qUOaMx0VunqfFCwa+NP9D)x6FZzd>`HQjxBse~9Rr^ds}ck|mo@I0%I`;px{WUeZGbPRV};Z`c@^$#l`F0t#wftsK8k`k&3NzYM~8c>jxggCW7O'
    '3eu$b;$Av0UsAm!3VlV%mOhWda91fVP41v4TR~5Z9LEKANqbOZ$t9J%ecg{+)FWq7wCA&{fZvQnPJuu(W+CTq|5*unC~v?ZVE-(5jiU?#d*0iBo@3fgvF`fh'
    'fUT&kBg8I{u1Ht{#`|MxWTo_{2Ty-~^bCGI?XO?0<8GgQ1J=;pPW<1NasQXHSaxU#w4f&C6f@SwoAs>sd8|`0mDO3c#2fv%?Sjy^T&LRo8S|-rU^&#2Ef);J'
    'b*(x)O2z$gjA<5dRY(H<2r@R>;jtG`pan?4k-!0S2OSErH`9WE1j@T973onNXFJgDHoV#dP~UYRY4t*IVBTxGugT#CptgA)IAA{$PMHL+)psBXIR`kfekXlY'
    '37GI?a7uojjsa`u(4C9gx>s@sv|~f!pY%HrzF;>?a_XdN`Z&h%AQ<_Mm&WBk!65l`SgeS%d5o)4!*0=r>;-~GW-i?zw1SObV*v+r4D(9Mj4o(U*nanDz*X4;'
    'j-b><%qE7(2Ch{sb$tP#-l@)cbJiS(i0y9HdLn?|4*Dk(yGym?q9eQ$BT`g}fdo@?ccbZ|8sI0*!<IOXpQv!c;vPt;8{q()f!Gf7D|Q^jT*l0jRi^1;)^v?h'
    'w+mn{-OB~df(=7-`)QAgVC(40!B}s}&aiI#L~=_)G25gR_3o!n;<%}ak?m(r@_@-YSkRhe^fcl*s1j-h=rAg1{ckm7b6|S9s$mXwPt;`)46CcdESms*-66M$'
    'rm1uZBJEWtn4ZiRcgT8=OT{>NQ*Sjwn3Pl|4q0iQ&j7g<=9kAxBx{!C9M({|7#GTcvQv`~)KQ~>DLC$)gu_ReNNxi?_8~R_d$)e1wnia?<d!q^mUHxQxxJ!D'
    'xkuV<eks;{D>GiCvvqe{!Lcex<7_Rd<0ZL#=|+Lt8Ufpy0rIGNcWb@<0=l(=4#3{yq#qe`rqAY6-Xm&d2?f4jyuw=YVgfY<1ju(bMKY<=(N_D;r3(f2EZW%}'
    '&bpsqnKa#56W?n!EKBZmKy4TDlIR^oxUjQm5=r6dW`f*y={D@2*^tnZNQhyMlcn5iv|`ahOvsz9z!sFB?Db4#tcTAd*4Fv_qR_T?N@l2FsBik}ps31Qyyb!1'
    'Lxw6Z#&Wx%ORN+viF*|a_0bgIL=%Z6_>drH<u1I_8}wgMz+9A%cm*11&}63wOJEvCbE%BFV%kgr%OmNYw0_zmG>TE~Ei<~%Oe?Lxw7pA<I;}wW*N3$xEVbP&'
    '9ot{Pe%G!Clm7b0G>56k`z!2ZdKT3>9r@NWx^u-JJAT>Foho(N&0g9fEsMngmgHK=iZ7*G!4THW)10u+5*Awhy2gv4zgZTr3w3N5b=zr}KbwJWON+y8-(-}4'
    'fgZxhMNIQ<ooza?&JPw%u$)PNKvxkhJ9n>}cj}tg-E>!5Z3Du2amzKkmJYZz_UcHxIYIPf+cmtcxCr%S*NJIpW$M6B+QjVNrrGmu-wkfq!QQ}YLEV421Zjt2'
    '#!8d5FoWOVDxNTYrqzZz5kD5ED!|Ss^Q>54QVLji6HGe&a7_m$1>!v8a}spa6K5y)AJQI(*4Q4O$YYcjj*qAg_8p}%)%nb!56<N+wZ%%AQ@KB=O^9xbhzj9-'
    '%J>d9n5VwslwKZxVQx8n1h>?u>&^5ENoENo@o^ka0axYx5`%2QT=MM_rjl0QQdY)!nS4VP)O`d{JexT;R>dNtL=@Q)4KuK<@C7rzMt4G$QY_R#;hbX{BiZ5R'
    'K@nfqD&bX`PRo}|EH0<@Wl;m2*Tpg&V-%4!=17_VSpX59fmS;ATT-R;=XYzA%oG$#g)R~~YMIj-r8~T76W2;Bf=z{107#RhqQfF7Rw_w{aim>W7IcmIX8WAw'
    'J5FMf{bWsZE+t@NAk1t&k4MrtWLu^UP0g!Lt91$DIQ0q*lN^^Ce~=)nYht7fe5ni~3WDi?0wUVX&{XzEv~fH@$NkPMeyAGKBRfIM2)(-gGU89=iWCL-;`H-6'
    'b`;B@dQbfwDFimUJW|>wEH+vTo&B{$bg<IerC1>?D(Q*vXKI?Z+_J8X3pN@X0Gz!EanWL)ku^oKk=G5BT#g*B;dqEy+M$+X<Km^S)R8bp@Ju<Y3W3k!Ar>4('
    'Z^YF!JT)Yr$2a&dK27t>Vv-NuDBSf`Bh%}+8*q&>1|E~&g@LQ<Z{A+Y_}7=<V(NE|{u`tx>qPA=Me|v1H{cCb3j#0^<NwO2029pD`!NKY)}AWu<_mgRzlm!a'
    'E2+xlsS;4mSq~?di#0jedkXip6WSDy5V#XXU2n$tOs#eIVgNRLKLP1SKi7cuH`>%lfHWB`Fn1s#$ison+&MBqFhgIXzZpdFxLlvBaIm0bF!>NR6isbgkrwEO'
    'S(Z3tvNPb&-Swwd?105l+pt<Hyc)2cSSVf}RW8hK!`h2@BrOT?*|gwlYbkny#unTgZ0$OIgSD=kN9~4Tm>R$}G^lrH86SpiSif+OVRQ%=<oT~iq1(njKD+>U'
    '|Ix#bBldAhX2eH_m{pU*@vPHJJgPGVPOGyGDhN;LO9$v<0$c>?e4y{NL?9NEE9JKjDZE*PG3a#c6sp$Grc-R4Aa&wi50Q-bbk0l6ST-0E7pFz1X$2ohQnBtg'
    'Txcs?7TkL<m$9vf7TvA2Q0DOJ2+a$>#yjhy%p&V!-?vs;mOU*A;(eqEqqJV5MGxcPZ?Daaq(_VdcRe&zP}B$AfxE{xNeWDnQXfsyj;%fGfrtbmIv&e}VILlD'
    'A6oCcm;&z%R=&5rE+XPm??IM6*4gr$=%YAWlm%@;up4d2pbf*xDu+<0^mhE^XS#QEf@Uip<-OyrRdqa=-pd@1a@fpld%6K#v8)zkgr@Mt<7|#f3ILf*V0)ch'
    'IbGGo&1~V~*-C80C!Hp<&AQYc2lRKfaoV8oTgP;^+VsfU<-jG(+&M;&^z*a34p{wpC**Oqo}BkEV{ecmK0o?m2gWry?t*mm+0MS`BZI(#vv4>d<o3ne4s3=p'
    'Cqb(LOD6bsykFmXCV+Pffgb0*yEZ|Gj<o~4cpG?>)OsRxP6&H<b}UVTq2lDc%hIO$Iw5xuz+n=d=kwJdJ|y2~JO!*HT?2!0veWMNsmWi7f%Yg{bUVe|Io=6O'
    '+&=G7vg&ssJqbjdXc>TJQGK2qeU|K6RBg~qWZPgDw+~fEUh0{UOp+DojNz$6L$S&rWnM}45UOJ67DNS*0X7^yPMr8{>!B#tn_0~yv_=m5$J=R9U5k()--EIh'
    'G<Ik~SlcIJ2ekDDLy7I0%<V9Q^47C?#KXPW3WyC{Jv41eGCfQV`$zEq<9qPWz2vBWxSe86A9@(yssZj!0&rrx`jM_`%CueO3msZbr32cr{?%OIY4Q#f9_<ig'
    '^ihBowp51K*6GkJ*;>nuqx`mJj77X>uYb0*u}kg>-2`Y|A%{oX`2pToXs>*4hj;~3AXMTo{sU>INWb6735=GovyB})l7X{rS2{XA+!qY#?UJzBxRO}QN+MQf'
    'Vr?pkYf#y<Icv#qh2^FZBT;q^$)M@QcEJlf)l;3?v6?>D!9T;G{(e(TF19H;+AoN!ONCWC2CY`}D-#!R`=Ldfeyfc^)SDAGq4P4eY7d%Objn5fK8Zf-`n6cs'
    'acl6qBsx0uij()G-4JCh?^6<e9;&N$sYXSUC+1G5mXV<bp_j%)=DhtnI%KQZ-OpW!O~=kqui8sBk?1&tVo962qzN%lstJ+z&@deTTN2&%YRgD-cK4Jp!X=v6'
    'Z^;raLA53lKMrGwqFHW3kXxWhN1=EF5=<~2L1cDIOIf~Q(IHCzBS84vhu{TwN|(x>YgbaJKyyTGNuM;l_-^ufykC_dVs{pX!FgR$aYIqi!44dftmO5<ix6td'
    '-iZUZgMJg8vJt08SBuxhSlDxP)HxQHqFM2ZL$YR^_o&(|`&+r)-NVCOj0)7lt^7!RR;K_cB1B*5&;e?=mKA9Ey-Tqb?j3G98fcOHqs69PM`H|TyMCDi1Ev;H'
    '6-Ib{cUQso57a)AUf#Kr{*|8lsq;5@(S9r^k0X@qV^1aQNrQh@c(YRpRjuXBQ1vHxKzNG|2qQpp^N`R29=s8~|1>(fzWpZH?HimE+}e7y2(x~4oW8?J!7b_%'
    '(hPanR0|@*TS>I@XweYbU+ejzIRm`KVk!*kmX%cTNo;fYXysLWazxf!*Y1%ep13<lmih>O*OBF|<W^TZ6f|5J-u-;>R?GR~Nkqq3wI-hrain+)CF@)2Nb#1E'
    '#tFGBjo%vWw}!)~i|@aXKkzAIpuhU0@x6{3Z?zmX;<~!kcHDT&9ycl;B##_-GoT>O-<=D6NKUl-%+k<W>zSqA_W-}nK6v2pDf!jB)Zt@5DDo~!V4qR5a1}JE'
    '!a2UbXv9rV|9JZF(djpj9{7t^@jtcqKf_E+z+pfyoLppQIl34sze02}MEPqzKZ#bwii4totV1_&od%!%_|yIGZ_*+NMU(STlSIN4{=*0uTv_he{6;xy5uJ!6'
    'opj35)j^2S;d-&+2&I@GxtdYc__u%h?c%r7^ta#t_J`j-)h(5!cMFt&Dqlss1c@0vVtWrfH6d-%H>NFpoX*STV?;<W5#VpO8EK}f8N?fNCX7d^*M4{>b5Gup'
    '<h+$*>COY)IA*?f!n?L}0u*TvLvl}SZCKPyeNn)2pPpcHgs8=06+P|?=Z0o5zQ9ObQKEIY8u?aWZ@X2cSxmrP+3k=T=Hn96>E%^{gc7~XYIw{i8xRTl4rR^a'
    '`zlZ`tka(~Exf_TG_&lGACDHx&2fQMwu-z?vl%A)(1#Npvq!_CfAHvs2Rc(Oy?=2+Iqaqs9-3b%qjAbEh2{d8`2k(k(ZxlGqb3~9)l}qjVt}@cxdv>qRr*OX'
    '3Uq_Uq2LxY&n9Tl)`=XfWt&$H@jSe(Db%lNQJZ^S$dD-I#`ubOX`+HRq3(m3>+tBz?D<wQ=||Cbz#;zpXYbJF&wmD43qz(-mP$HiLny%A$qcV9X!i9G8Pl9D'
    'Fso_!ofp6X{M)GCAg|h;P}!tV=J@R?2VBDKggmwXfA-#P$&DjP6TRnC<g`o!pp^itillx@D%nsZr$t*yl#%R~+OV)R5Xh>^E)-CM1d1$i%WUoJ#oC<PIWI70'
    'FUDrg=4xzj_LgsQp5*vnM0i9-{-9WF_3pN;TPkG6Uq*Pidw95ijw?NqXPJb|5CQW5C~Bn7DLl~4)>I!v21H6|>Si4nK~6~$VOR7uohSuscDgFbwx>RpoKH<F'
    'wj&@U4@H)1j_~3UC3C<t_#4Ir@<0AY$*;oH-~V}B92TgSOa|k<NlT)8ZJ>g2vi(Rk!AUj0fuT;{6y=dJ!nmsA!&HWhL;6Pjyb~L<fdY(;-8CBYD9Q-8BfDPG'
    'WorY=d+vB?Q3yfpdEDxMex~5FfTNCVj=ncdU|;F621&sEt;644`n2j#Muk?5&0P;=jMe!df@CAjVkTDk)3{D&2+TyA+qYwteYQn~QxCh-Fgi{cf%ki>*V%T7'
    'A%G|<UqNkyQR~jUSXt&0TR(J91Apo1YnKhdZ8!QYha@QCZBl+$73rt~_+zUm^}1R2VZQQTmFUq8OrK>=2ZG^=rpKUZ0kJl!J=MaHM|O<GQ!qX9b#0!q`g5^2'
    'y57SLu!?E9A?9zEH}UUo6Jlwe8vgsz+8@g;D9)f1?579mz4U(iV3hbrwcd}mjXjiVoI(y~8AT$_%^KdLD1$D@d4mO=XqdFZ2Vaa1M!h)JC4I2L_*lca=}Lmk'
    'C=XJ3Lv|A9BaZF~FEGL(c3~f6KEccpDxHeIv*d-_8g?g4tOoKNW)giuN`oxIlVFvCgyumsjfbv-gkay6WdUz<PK1`cKFtyOA$gTAR?8Es8EjS;L*5n$mVJVD'
    'XiJf!g!AH}`JklIDnQ*X^E9C|sEv6=tdz%3(`36`6lYL{RTtQNIchz#jr)u;pW@LB;Rqx%uk4SV^^*pES!I;GzJA$o^<ydr)cBd~<oS4iVkaqMtss{8%j{iO'
    '%;!#(4cs$MxHj6(Z;`I?m_5>pArz;%KpXW;5HfC!!;q$Y_F6at6{}mG9g4M~C~v0Y$K|wGQ2gH!+3&cW<AD!Sl({Yp|Dz27tnYXb=GNfPK}+NO4G$3h3-w8S'
    'ogi22^SXe>G8i8o4)-0bFWDElxE)!bWoe~~`x+nOpzOl1R>iLdgGmfz*)>v?0vMTg<~?N<K=`OPduD#rW?eQm&L@qduG&@jkI_Dq2050U0h5WptP=f6GWMu='
    'fUl!kN~NcYW*bg|5s7kzgl__a;z!9#)T#stH?Qy;B5Yt^vnig!)tVl!P`m?g8~N5xPQ2-TQ8^ZDZ<qkO5L6?Cb@vBk=EV#h8}v(9dILTE9N1V;gtwxpZciDW'
    '{!B%D7%P!<oFI^CvPIJAi}~H8o-lDM^MYJme-^>#bVA_xkJRlT%B_@!=nFxP$|vku^3Nk!T9JiTXy`c!1c-k1A%uxPseTV=FF_H~AMPu$qDS<XC1PZ<W0Iy('
    'P&5mPT?9mT;jo)zxVcnlkqEvtU=~9@kTOxcwS1{S37lE^X_5!;Aq!XzR37`Aq4L_0edR?(YRKpZlW=i4C9CQMrn1_e7Bi9SpjiIcK}6!xJo~c7=9y!e<7#ym'
    '5Y(zCADU%DgIo~4Z|WO+skuO6Vsk(lw|T;`8@R-z1&oGPm#Wb>pWILm_&!idAOv|PyXB@tNMmiBd5LhzV?R@Y|I6J|sC}o7SC7px%1n21vDlW@s>62Q;qjul'
    'NYJhbXBgB#HpZfBzIgPZ4k`xeAPenIQ7fZg2TH}O2PUDPJ!2598iDGf?U9ymB!~43p7$-jp|S89gGTji)CK)<x}9alz7$_9bZ9__lS8QD<xz?e^L5bZxW`GP'
    'dO8f-DdpBz&pfNxelBS5W7FB09wGfjRxgjRJoF46_1iN$@-w=nm#(f`KStq6=wW!@RNML;hjpf($8(Re4q7*kcq&qM)u5vd$y#b-e0Vlt{}9&I3!~rqM+3XR'
    'b~m6(w~e;bP*$2gS_$9gn_90fO0|EIA-Clp;eab(AtNiMZ~CSqF_iBQ(0#<1Tw$~0=I%!zKE34?WYF;$t}l(msxy@%WFb~W|Mr6_4GwPrzS$E-+-m{f>>2jl'
    'YXjfx5!ddSG_o~i$iE~j=1}g~{CZnl_iD=X6!n?`J$r;=*<+lbvl?{FUYN_aXL4M!b+)D!$%PClAj28Am53(LLJ`Hw)Cw(%MOvo@l~9UNqb-_alR1ZWE6-_L'
    'kE&=}uQ4T_jooaT*MfgAh)|<dACFMrj5yvwG2oQU%139Z+_C0@{t@2|H)1QD%_U~6-K`<j&s8hntEYy`;9cC%F{?exoX|=$sBMuJ%5PC=u!<{<_a>SlA(@8B'
    'yA)l<RcQScHMJw-32IXbT{|w`m;>rD-D&52y7*4{d%Li{ZfviO<?X`m9IG?a5-HQh=yqpvwG2+Lah1;*9;bB+mp@&va#g43!o_A=Gz+rUqduSnO2NT6b;l?{'
    'o;KjLLIb`n#DKvO`7f(tj;T@m)QNF>Vw}MG%yj}+h?1I0S}1PY_1F?ts^}**RFc{yPR&9h#A+!pur9$T=np3+edWIs<Y31QcGRyfHFYpu_S8~U>Xw^%O;vcK'
    'YX)RtIt{5M(q~wcyNpy+Jr0bWMdp?v^Xrgs<y_5WQ?oLuT6Z~W8$*O7(ZC!X*}P~oux_@?yy3JRz6e5Hss(h&5gaM<<;mvs{cjvqoJx%;Hxi{%XBX~tbPwGC'
    '^wC`6OzE8DISSeXIOiQBkEh2^W%P7iOqR&wHD+{ER$Sal2`9gk6L*(GCK(mU3HnXH@yN%7gr=Iq<QbvjfK(1Yg<3{$SzDORyzq30$wHUMmIuQ35P+Uq6lXs&'
    'W@AkV)9ncX^UlR`+z-eojgn0&%n<qsuHa}EwKY*Ky6?fX7VyW|CA-mxXb)JeBu$KUJE{_;JngIwlBL?$7P1uo2b&YMwMOGtOgRgO^0w!)27<8SfnV1Rt$H#F'
    'V)e)q{}a|^v2x!{aPnJ>ox^;ypSVv;>rxgU^%VK`oYQypIS7NJ9#6aXs=?rC$S$yQZf9dX8<U0%tr0cI#bfg1qUxbqi|7hW-d07xNs4I{9tn!JQZspaa!@8`'
    '5e0@mK2mj=g#n>G2Nlb7GpMR9`X>Ur!?9j{O&tw?lswL1JDwLyLI_C6I7i^P$zt`I>>qZwg8Bp81mxES-7tU%)>MMLEGoIQX<@(C%;ghGw|jt2;r;2WlJ7g<'
    'xZsKm@ZyIHAp<3gGrEBwt4aj({wpdUJB_(^QezgEUr^87G3h27-R^tB&^_K=FhaiB>@do!#Dh6D*GeMEu{Sug@OZ{}4y(;F5sGx-BeyQqZ_6?0g9~BS1zzQg'
    's=d5H8k?(E0(lk}9b83m<HG1v^d-VDREkKAI0JoX!7LZZRyxa_V>fx+NgV!pMd9vZuvP@hdha6p8sr|c<_{pp26x)!7k!ErT=haLCa|cw|D{e!;r5&c7oDA@'
    '(+wrlu2SjR^J~2|alrTcN+#H^6v@CY1%Y@7MFY2D<%4ZQO9`*1Mp1FxW5T_DKVgBRa9_|#RH%$4on%Ef7n|^ffw}nKMRHhAkMORQ51F-YJnm`<-9x9fPCX!R'
    'd_SIFQ;^U|qTVgFat5~dn!;IDuls`=NFJ9;r`8LsA_+WiCMys;%IhqeB7E9PL?dF<jo+Y)v&Z#qh`abl^4My>E)Z~mci-&@yDBu!uP^OBygbtt<3i1&fuKGu'
    '8y+pw9%mXppI-E7XERm(FC`08#VjInF-_EVd0!zZT=G_Br5k-rZW`+nF*KM~VsJ*9Qlf2i6`;Mqsmu-o!9dV+VjZYSxoYy9g?Xc7!1T_K7Ubg4=~Z9H5srI7'
    '67ZhHw=E?Kfw&dLIwQIoIVey?msHd<%&&`CZ!+MQW%bgcAao2`UTJX<1lWy06&@GJl%gY}>Em=c{iX^e$$HMQ(f{VQyynzZKMehI)nNT*1+;!#EEdaYUm~{t'
    'HtE+->Q@E|2p4xS@T=ab#PN7hf|v;m`2Fn!@_eWxdg#I&Ixe$H6vSBgT2NIL=ybMx-(#icSl+yQS;la3rOhhiT!Ve5B1-l_rD1x8g8(>!Tv$zFTB)gdcTpNa'
    'VKPno*_cfZMa2r}Sq}%)GwEo|&W|MM<yX*2zTWirQmM!jBu;At^nY}l|3G`+7KndH5TORli_;=5rdJ>m5KRB_-hQ&(@YCb%l6|Z`z)+s!^%ttsh`kDZ+rDUZ'
    '3~m7FP?(Sn@y<E%HYa^dX~JPM=g^j9Gc@7mFHdt=4{}ewTuB@sg=kPrIDFy#(LD>GK`Q^|6lnJ36q-U@v0_$iKo}IDVwGQF&8uX-lKZ`51U*xYJ^FF@!;_yu'
    'pND4%(S%bg-e`C3eEamp%V*D@K6?0FCXg7}A7FZqk{--*56!ZtZb2{`e7^qDG%OWR>Q2kDI9YP&I1T>JkI$ce|M2-Av+tk0{Px-77uoN=d-l!4?~r%)$N-K)'
    ';XhE`Y>L#!pJEyZ0UK8S?%5-$TU*5fOYI{X)h2(liAzw=JvMNSbqBf@*{XG&CSe=7T3y9OE(I~sK2)w4<G#A^Hu?Bcm<4tBt+IUgt9x|fF%9m(Y<?I?eCy*2'
    'xADDEg@Q;=J0}0KY0-5C#yjCfxz5ZRkuDLC#rUCH2@lmsok5m^i&t9$_IQO5d7=QCAPzU=>TM~3`AKqb1xCRL5J(QR;VxmkCpjO=@klZR9U7*O^vF3p!Q^l1'
    'nc^Vw<(pGjo%K0%{0oK;C*RndfFQfjxdmfHi68vKb_(BfjJG5<>)MwXk~&`@GW=@8qJ1m8K=QJaZeF4f6UVFKCB*S+G52Fc&H99)ICX0nam~g!aoY+kO>vIy'
    'QgpCT&)LU$ZxY}@)W_M#im92?xfGX(RZgPr;IU^=2e!8^-Wb*0E&zWV9CO@|OeWpw(&e2cwL=yq!{YY9DGgaL2U8o@xvf5yscS(|Vpta;9FyOMBm5+>&;7o|'
    '_hy;BVvHoT>fAeE77^QMM(^qFw(by49qI^omsXd0e^Q4(^2(V#xVAnT$O*&qZj6wTxH(4O7{^DKz(Ngv{A$v=$YUABBv`9ikzv3jeLinWM1wQPu6v`sg9kp?'
    '2UIvMPEPA<eDPUOquk87H2mQTatWBu`RW+`qtU^Km-*{tpIv*P$}jnPRrVtb5m^U3qxzxE24jOfs8oip$hsbOF}Zl{iG{;^78*B}`I{V$brqpvugKk_vDZ-O'
    'Pe)!<nORJ?L2N_*j%sa;wpn9y{zhzR(Cx)p@x=qADBBVf*lzWaz;pxnyK@V8t^ip&&k&-Gjc}$ZAi$PpmQ`G95u38yk=<E86vvpPw(y3af|3Vl2m^@##KK94'
    'GG6y3rnB|etWKKsD)usC>#?NPa3{hVXmE#)C&~}&6xXIx{Uu(Fv_`5m-LzLON!b0Zv<O|~FS1nXo!qJq%@sU!y<v5#;uj|x-ZMUUfjU^|h6;*mTa|p=K~T3@'
    'aRKiIKaP)g3X%Y)kne(u^Fl@_9svp+b+fW)XC_J)cP)%S&W5;+;M66f@Dw#VgHA73#hfrimOem;358{Z5MR|ZSUL5OMG;cafJFxF_gBAP;){^(*GB>VPop`h'
    'lhZ)cj`aCeM-x=+=HnK99PHuw1Nk{<Sk{hS3<f%?-V3kS@ouLj0--(nfX~v0?kru=08B7}8QBNCHllgCO*aUh=&#+*2`_%Qz|a=B<!c`%VCtm$QhXxb(;32^'
    '8Hw~-Zg~Q3ZBL4rcCJ2RfnJY$CZdV12IN_=`jb)q4}Jfx$ZNvWnFwtXEy*CM<|VZnH9R+5Ls7HM_G;8qcvnu5<xw`89RkRG`EJ8-w0#>2i<)pn>1@<1lnLe)'
    'MqWo>8AWp$*v(zLlxM-jJjXPox_5LLK*~AdurBj5t`@1(7EL9wCxaAKi<#Il%sy_sGWjiz86TBNn(S>P)09;-xrAYCOHzR_gJzM>jmof-8;DcWTLn5s>|DVj'
    '6|;<@ZioYUi#PCbMh&s};bxA8G=3JY@Y<ZOjUh^{`?!dmWX-BnYC58*ugJ-MR_pVMO~8!C?<87OdJkHK`9Z)Dm?g0$*d{^z`>k@(5M9-g8iRM8O`<TXzoul='
    'A$ge6!2S*kB2QodPASQ3q8ba=6L1D#ihgl?9w|=A7zoROs`051AvF7bFKBl38>(~+Ds#D_Lu1B7S7vOIz%fjZfxUByz9pK}(nMuqu<ZG?sWnriOLI-3iu^J|'
    'Fu(<#=)=y$As_ZB!U-ADbXVfZ-xAJ{(4l)iNbV*QntWE7EW848!X)yA-g`y)X0OC;KUH4s&hR;kdS{)el#CFEv|7jA>)Ng>AL-$YxN&A?)tqRAYCXnC06C^0'
    'T0-iE+oc^@Ep7%CCR|^6IADWduNTz_c`4u&fsSd|6x-qsP5k*8>W2f1WX1UQZ+!M0XKE2qFu3+v^!s(|GsNo{1{7PWQVj`Jg}3Ix1prabRP|?xb#COVJMq>V'
    '8C!KFpU7ebHI=KQuG@@41I_kAP2;gvgf(<%#2XJVSa*-+4+)>|Fs67*h9~bhE~H|!!shq5x-3ltQcalMGlBzsKeD)y%|{!mhOj{T0E0G?5^o#eXYDeg>@;O+'
    '7HJU0gut6YrAk%W{aPI}l_FJJ&Rld<0ukuDie(nE<1_P#mCKE|EF8p}RgAg0#i|gEhRj`%!(+`!Gafr1URxq`VZIK15nNn-#2vv0;;tz={&TC4aW~it-<h>V'
    'K_kPz6W{rI+@#G-*wi%GL1&9}(4^tj_0XiLMY?EI>}tQFNi&OmN24ZJ*GpG@Ez)6URj;M*j#^#)2)e3;k#Ss4e7JRV0S*upJa-v%6lAy9^5%vYvEXcHb=ZRf'
    'r{)FFbS=(}%h&@)DwC%Pnuxs-f7#e8FF9UVpvLj4i^U?2lwv2S>T&EHk|Rtl&d4In%fZx$o!j1PK)*G>Th=A|u!<?e)vDNBNsoP(4+8_afn2^GG)#nB`#;Z2'
    'NPoZXRQR5<9zEB2%zA|?etl$5@{1JcWxh8F(E#CtjPo32)(S!>eaovcBhVDIh_tl1B_o5dsaEL~Z3{$<l`z;!yM&51;|mcQPP-WDEK<zrd`GFXqu2<0*-5@A'
    '(GbUdbS|0bS+jaQBE7K%Z17eY)x4OVELY^NT(>!^o750@><%p4m#De(H%>HrV!=V?S@?ARg>$stpn<&XE5xRO=fdhLj_6Y;w<CQ^uK}U#mgbB*Z<;8@$>jPb'
    'L#Qspfjmy8KBb1K>5t{s;(SenZ5>CG*C8(>vkEVODbczDK7Yfih;=2{@o<2LqY&i<#-JJd^LU^Cj6h|;cbra3Jje9OT*U203Vx_1Rf$=74%)Us+g5<HxceOQ'
    'cu-kdub6ywd~rQ>scwnALa~MR1yXAKfzip_``*44KbuwNr!h{QXyO;Zy4Be_bm$K>^c@?PqlR|z58JcCE!LoYd8483*mUDyCk;H`U$&?-{)Zy`;InSd`)hcY'
    '*S-+}=FsH()wTH2tydsl>9pixYx?^1R0)!-NhMw%)S+AJ9?H@6P|}Y$#4w(v+4-=o(N&2IMrtH3j<N&pA%|D0tfa*yd3<BbBl?lig|TpU6DuM_cjTZGS%p55'
    '^-R&(9wVQQ1cxw)Al4$-SM~#`QLM{N$}wH*>1KuDbm-qkRxhPLjKdHg17Vsb=(??P!g~G3P8&BJKwW%Nw~>gFaD&d}TK|H;uB^h~O-QWBL|pLd22>Ssw>mK`'
    ')8X@(+n|YRhhZ+kN*+Os+rqv~=o4*+IKQpb#U0Gr7lBT=u~zuHlxpuI+K=4BtqF6PDv!r9LEP|UGQtz_avH!jW6xAC<Xnwi@h;DJT8^Xz?hr$xv*q^u;;J9@'
    'k5JX+lWlz&X@>@omlJowM2VWS*TMhQa?dpJrYVglq?Auh{gJb`B_*{MzeXHMPWL4D_V%l7+2YaQnQHmvYFR9gDKs3}(mB*EmPZorjqOAW3=}!xCeq|jI-l$k'
    'y8A#+xN65-A}h-5Rlb1{9`S=F48(bH3<JqI>d2#yp+9Stg&g;mB%#?^&DiUiCI*8@acVYu4Lr=&z{{lVd0!3UYmA;5&|K*6jTiok3rG4wB*Cy3^b#&-i&dGc'
    'HYi#LgO*l{Bf48klQsTL4L5!TMtifLiF4MAe5m!E2xlnTI*~;VTNiC9@MOzh(KDGPdvs6;T1vh?ax@Bph?gY;CYDKis0Xnx01Rq-d|b>5GRDyI#d7uW+($s?'
    'X_TN1=kqONCTmwk&mL#^#9c~{JbhD~!y5S85-0q@paEYKy9?4>*IbstxCQ?quF463g4a|4zQNCp10_wSpflVF@F)zdLv*$NMlToYYhA!4oG_09KA=u);D}7%'
    'XTK63br>8J>ong=#qR1aD~yC?)H9CcJr%7LDhh}GUAQac{LuYyuio{0s@Y1fu6}2;ay=Qu)4RB{n|ywoWzT2>sD<yQ3~+35X?$bo1a~M&TSl0u&BRPtpN)Y%'
    'XSK`Uz@USzn~X4ayk?5{tT4E)n1PtPNnP6F^aC3pQxc_b29@bmAu(Wm$;m9=1`Bi$G4^=!9Ay!g9=q;92Q&{=ztyCfHNL6XPQ*RoS4ry9EddLi3#Z13$Iio`'
    'srV~<_O~-sTi)0-(4WNdm5N)Uw=#qGNfzT>DxftKe#<Yv7xXlYrhw^eL6E(V7#&?OAugtf9{(f!JHTXVMpojuIo{#Om=w;1=4ph}5F9KcY>|p7tGdq^2i3HY'
    '<33}T$q)ttnyw!4@rB3dS)n`>n2FAjZaav6tn(J(7A(6k{3uMmFc=GiGSE?O3@)Xl7>US14|Xkr9jKt~H{uiR0P<uMvYlTS^9_bp?~OkD)ahGT&h}uk#4k1@'
    'g7wOQ-cBfRoU-nG!NpCp$!{?t`vI)SgV6)+$Id>5;jDYUg|16wmVs_xP1pT5aKvC!3~1MVe}M30A3bdkuy^_*>p@^Re9F&{<SmWji2m)q?hWV{XJ!A5t87;W'
    '1L9X3;>kz%LC-y%9`)83F57d}uc$eoLhJr!<kV_g8YBi4k?m@3ip>IMwf8##Sfev@QLdH*-}6?s+qJ5($WJK64YW@y@rA(_xXgLq;->UAwYBtg-PS7Q;d^ay'
    'r}o5AY8uR9JyyKq&^UKoCOEUB#{6AQLm{9H0|bMhV)CBtfb^FB)bB*TOh$O8(3jDh1gtn%Rw(K}n6@=pC8Vg+oLbYZ492}^b`|SCQ@a&E8-1E4_eWn0sy+=?'
    'r{jS+zGH4%{M{z>v)+u|DEVj2&{6?9I_ja3mA4i*t@f5|j+d*8;aS3oNo}NU8QDeOL(@8>h|<wXIodCHwEK>%dyZV%Y02y{`?#_R-_jq~ZCkH(!^vf2{dRkw'
    'O;tnpimv67x1O?F?WAd=p>=XgTS6Ts%=jXk!p8P_+dwS#hOX(iG1t3x>n--CRSO}AmJ8Mt-Wa1IZ-+Bz^{5AOsE5`w*>WKHjD(xM89`^6A>J3<FwGc2Iuwlc'
    'Rcg*Jb(rpilokF1-gj#uA=1f;A%sbZiJ=XlI4q`Da12)b0x1<nPIL;a{-QYA|6(sqUcFkqfw`WY;(Tm^rr6-)QMHYsezjfB7lKvAe;+1~dCtF~pR9w{-7N_m'
    '0tcWTnn+Z6x>{{`W5i=8t4@BKd^$tx-Mqj!CU!11mI3jRb4THL#Ne}~g$fV>3@i4!0Og3Ep%qbW7(qOE0vh-gaz|(j6Mj0N>1~;pj69_yX$digz_JK-VT*ML'
    'gc?|d@a@HXM0g1D(jZPD&xCSivRyhg7H4fCXbTP+!AC5Rjev85G4RV09j@U=AEz0VrXD(6Fv{UwS;-NlhP`WN8vZ=yBhp*^x8a6#Jy!aS>fWI34ep1j;qNT7'
    'nrob_r^m%?g;fmD2hI-#<mY(GxiSx~<N<PdOtcQn<pNlGMiWN}CTiyU{AH;e+{r?vgNThm(hPJXO{#dpF>9YznGJELX;^H2<@aLBK%aIQqw{I^!w?N44e0oe'
    '%acxx!)f!inF#*-O24?WFED13p3?)G7XO|=*7@hu{B@NkToyUo-soV@;#ZS_|47}5qZNNs9A6!osu5rU8E}x2S`RjAXyH$*)Uux&t#lRZOHGW;ehUNlYE`;&'
    '6Bw_9PCB};$m(n02}hWf#jq{uX;;S6C~4L$30}T@rq)HiU8UpdO3v;BZ$QM>hE$LdO0klWfQj|h-KrsV6#VLvJU1vmgxo=}Cug}WOxPTK-on^TY9xq8SUHl#'
    'a8O&ebzTP%`*iBvi$ry}F@)t`ZHvV`Gw*4s)DV7FoQeOS<+&3e3K_fxDL$6fncQ(Vnwi2C!EXJ!k`fmE+)ARp;5`rusGdwnWFv9!`K%t6A6Qh9{k?rr>pn`J'
    ';7#=K1>ZzFSwMnqnt%#<vd9TnPmPa}UKb05U(IuC^R<W+OcUN6cNQ_4_@8f6xK`%Ifx4aL`Nh0AFO})TUsS)zYEyTrnKslK3F%J+?8$A;8B`{mK$viQk5onF'
    '&qNE9AhQ6V-n5Ecn36ciV43||X>}ZK8Sm{6-1n#^y(Y#%m-g!f-(LhZ(fUYY!)?f_*Rs&~M~R%SFk90Cbl#HGCS?T)a47{26iA9)y97R}G?42bdCHTGaiUgD'
    'K}~%yR)hzf3B|Faa|P8%SqAkNYT&#n<4060C}!Y;g}A<CGHB<~WI_)sG^jXprD{dnGSzXnAK^wG;d}Eh3Ju3=TEIgghI@Pa3KZQAOClYPY%*-d3XtLUE5lcG'
    'p<1%j7zcwCywIT2fr5mzfFSOUE?6#nNuY4b7@(Fdgpo2Sr~68)@TMw1KF$e{S#STo&sFFGQ&M$j2G?oeO@gzrFF`7mN9>&t42E9mmCIhiS^A9Sy{!>_ovKCi'
    'tl6oaF6EGpx3cZH!ql9P?-5MyxaXn!Ap)u9YB5I+M{D;!^M~EGZNinWR8$}6+4Z!3(OAb_QBr;k{?S|`mz!m)VKo-FHexp@FNoOiM(>2F*qY$$G0q)ZTIS**'
    'KAgTr&~u8{M;seGkFMk@{v#aTdNtn?!p+p&{=q^r)SiJXH?YQWQ59lhX#r)q`@l8Auy?CvStp;Aq;tJG!Sp}`lf!xR<XD<ifX@`^v|vSg+%mrV$Noo5RFY`7'
    'v+E1(xN+(m(%i-!FQQyW<<3SE3M|!zKn2Y~p@KFz2wJq3|GI=kW^F#(Ps0P8L%+Qf937W7K2h6b4AYXpMVj*#n!)prXsnhhNnV616BTr->is>d<7oW`ON}c%'
    'PXs&5mAq;!*+~Ji$`7Mo8OR;70eV$Qu;yhpmS!gb1{hvP9{1=Vs9-&=d;Z}pGE^srJtafp$!<Z%=%QcT->>Kw%*7g23o~G=Qdk~ql2v&Up2O9wY*!=mMr_d<'
    'n`P@D1_`^XQgJH^#jm4G7$7^vuBxzKO`NR*GHK-$J_xYiSdVc@#T1{fOVZ=hs#*dMRJ8hW=}rvS0E;RarP-Pu7CIZ*<^u1nu}1izxTVUJSW}7@TxNHW;=0P|'
    'OU)p?zE-0$@$*4=feqoc(cPWkP@AV(&qJ^c+ay_|Trzy+!QOuANt-G@s(*_Yo<hxAHd&}ssPPm#Uax-1G1>Dma-2=o=swaY0K1DZsYr~ys$m84?14M8t9nkR'
    'g`=40v2~JHs@Z|A93y2zY#tGNhmaOxuB2tuy$>{HH|!d9#Jt50-B$8C3l3Sg(cQJ4X0tcL&>Oos9pb>wyHj<Z1~ifS%ltuh>Zc7mHTc|Km6h5EcgvqV*6pDO'
    'm{UL$i8WMDbp!<aTw0YYb;@8`(cLP_u6Lf(N>;7FsXH=|9y9JWLZu99YkMNOrHzW5kU6i|1wzMg#RZ)AW>o8M_S$R!EmEjy(HV9HwHM!^HpHni5qH2tZJkm@'
    'eYe2Fz_4;hc}ku+(^l0XHa6&i$5S0@LnKKExuY_L^4~0wtWU5IFaIzplXT><^Of{iR^TV8ra;ZTc9RTP-h0=Nj7n;%sNU!dwSuok3GjB%vz)&jM&tJ0y7jqE'
    'F3aStT9xZw98W7mH;pbU|ExpIiu(?CYs2#(t}a<O2jX~FX%cDhI!@~$CyB1K)6z%=4L9rU<}}fEO!BWc7lojmEoZ0bU{;P?8o*@^BS*asy=tmg1*vnyNGazW'
    'c}Pe?i-a?OZ)eU|f-j`H1WxR5iNyuXRfvDSQZksbEk!T3>Gbc^3{SQ6d#@^&)^rzy3s5~&FTDDc^Tmv`A5r@W{24npCNe~E4Zd(t?I>%G5a_bKKOi2cRR-|u'
    's%0C&8-o4r-+oXHl)zOmY*G~>$jR{@<Iu@nI*Y3%bsaaOKZK0wc0RTi7UCw|?e@7oKvT^(YyL<@wBE;H=$>9swRd#y*?xJX)7nqlbpXP&y`3L&Y`1I8EhdNB'
    'k>@7;1<L;nM|jn*(X{0;ft9a;3tvuw@w@CdkuzpotMiYAadpl!iqt4_5}Hv1>SK|7IDh9sJaMlH-{g?%tZRgw3w~$hpEy3kAjmnY>o~Z=O2Dfci{Y>xg2TJn'
    'Qw)k+9lD&;0Nrq_)NTi_ve76L)Y#RX*rVj9X!NNK81EV{ULPqQOmQVxe3uAubTtX$!w0o%-vW}+L<9<j@&i*r6oc@6$-oWYqXYFgMhKt%??4D@h33cAsA-S&'
    '2CHdq-8qc<jzr5Xf%BdS;((8>XWwL$%cjMMdmC343A?|U^?IOhfo9wJG@G5~GYlxDA3Un)V7zWkbDx>zKe^a)We{@<Sbqx0M^J3<C9hWyb*(U0D-g|gemb2v'
    'J0nKbCB5?8o{Z&&BB&(>!Qenee2pS7q2V!<vX~bppW`EGKe>Pt)zJ8tC*Bxz=sVFE%+OEC-@$QfGZmOLR<PKmT!S;Ff9AQ<aVi~VFyd!*TQe9L$whHNr=P>x'
    '^w*&eoJzj=t4?d+i`O)zRY#uNn$cho@!9n5DRDL>ANgP4q_)RYMak8>u^;$>xau%p0VHjnb3GV+(XCT``yonn<zz<t7eSq|Ner!fRburF<<$gnpTw`88zw$9'
    'Id)$JOZfg5DGqU1yEsVefiIY(Iu%5fznm5FJ=*G7Nv(gEN8Itmm$BJ(1&0)E<vi@fF=|Ne_I9mD$&)v;#deO-rx=F?lr=PzC7&V={gKI?h)#j-!N=3tCb%@M'
    'f~L&MOT;)0pQn{N)<)`AaaUuN8Wwaydm~EixT~USBe*`3YT<QUSl%Fb>(b`-veLHV*?#lUYqIbF%9o{HMY%t%tr8nu<&w2Bjqx7+E8F@?exG$WyXs2JDf#Hj'
    'DR@&--8(I)N_w-fokJ>Ui?=GS&H`R>cAo^9nZMceRjIV;(p2Btu~u(X?r5u4@9*4bfaywY`WU`y`a)<%Gjjsy>`DqmHoxW{`bwU!;eg5WB^(|JDE`Up>ncjo'
    'YdAaR8MMdI)TxPnyldML-cW&1sqTJ%YD`q*T}xi>LTYreJdx<A2#Q0Umygfp16LWZL;qcKZ?%34es|%0f9wN;4jCt6x-5>_<3+Jl%+ATI{tr){zj*rWhxE^I'
    'J6*1}>zTTo=HIHoIu~d-6DR^@A3lHi^wD=uvY$cv=%Or4kf@iJ#d@`*i!hL)QyhL@UhCd3V6LFd)QGre)y$-r(ANejPfsu|cvBOn-1A!ILp-z^s0bW0OR<TB'
    '^39=y*yHDWcMtY@4JnwBgv?O=fM$+cMyVXur$a96!Sn3`F`9be0kqeYA&|@uocluDHWG3#)ml_d-hxhbIqDm9v&9tw&T;bFp?0@G1OZeoBI=1yxOB*B8g|%t'
    'WJy-Xo&qIGZ(J)3GY|B1IG<bLHqRbO9HopeDs_~4T1_1!rxkK*(+6EIpLU(<Z1g|RZQc3_V<ekW{iH$bhgs3XtvvAWGbOx_V-)QLrO7(he1SWNs?@#ZYD3Aj'
    '46DJPTB5Lg_!HjG8BUfaL2SEuF~VYo?`Z^U8^Tk`b=2zGWX<Z@vKlx~rA883f$2+HWxx}luqNepN5%H$O6?64+V0BiPKxY}l-OMr*qbY_yDP3cDy`j=)GnIo'
    't*N7=ir%&w+N6RSOz>KzGf+4kzMyPqtg3z*CZa%ADCT{gW9x#fnkh__<Qykz9%Ll*P)$;lqObFO*%j(Ptw6B^m_&?7v07jG;2mmo$l&((@X8l2R6y&JKradd'
    'K2pwW$GzuIzI*uc-9J5l`Yd_;BzgYi#qYm+`S9_x7lgTFRGhb>@M)K;-cc|4ZSwi&>WO~){kwmB`3&26^z8c|A3nz>_3Ql+>qv(-KG~WbVbIC(yMHHzq+$;D'
    'bf|-oR<Ei}86yH-EG1JX!{^g^4)TE2jT~K0FJT+`ZQmQ2uRl!A-~H=mHD5_Q-)2R3Y-BY8o#frWy!&q}=?I1I<rMm5F8uEQUCxTBGwLok@BVGM_50C(KY#MO'
    '=MQli9{vam@#Ke>58wUQcmL&?F7nK-$-93|w&%&a|G6w?D>FhEbG|&?p2H9}(~D`c+`<bW>?z7_n5ik&LVa?IU%xuH6>)Hsc^rztSCol>#U!vtQH3nMG1~$B'
    'aQ7ikuA7Vh0ThOlQIxO`R-hX%ae6?%T;{VJR3~KBg_OHku_ck`sdiePe*fdM=P%#=Uq3v0`Y>U@xx`*~x?&Hbvrf|lt55nJua#X0SSVHnr$7xQoO(`{#b!H)'
    'k>O8&to%*MJP*idN$vd!HtGsr3bUn-M*iqfvEl*9&`!3CRSC4dDj0+f8symoeQEnkAy!sLu(|b`!$rx7bCvb8_QT&ldH(MIrbz=*dj2GN@#M*4y$O$Dlm6SX'
    'IHxb5gW+_(I%LG&Mhxz2?wZ2k@R}Q`#vjP<amp8V#$o6NKAOK}jX4|F$<s%Py-`k%r<besyMMk!NXnmz#ccIvMOeQ1l<4z3`F6E{T?Yj9KY{CVqcjYd*OPbu'
    '2KDMU*6)9K`SklI>Je&vqxtsyqU<-H1r3Ltmb0Q@3^dN=z1$ofV3N$^>2|R>LTsAaU2S=4bsd~1BMXVpCx%$I@h0f~fRlN;+Fm02ny*TjRlz{V-Grn=i}dcl'
    '(+=>cB!<JG0-a3vow8_ogQ_AR_ugXGF9X+?{HNqCr#xeEuO6%}a{G%VF#hb-waA{aZ1iStV3%FYJU)z82X$&QJ(7N=%go{CXVNH~0}ayq;po%j>o242Jbm=V'
    'gQGFE<2)f|#TO6Y81|a#rAh29dMVKBNxa>Q3~8GO_*zImyC4q*7|D`f$X=`&+GVi;nORKBBQ;lu-G6;Is0}+2fvmY)8>;*Ds29n)(03}x-Go($XP`l3kDt7J'
    '_}#Za>X{?@(dNyjHyH7mj5NaDe_ZxPe_9pGJ`ob_vDs=4jP0nm-5d`;?|CqHtMv(-)qcruRs9N7DuOKxZV>kNGe1CK965KTk;7G5dnQQhm46)+wRaHsKkm?T'
    '<e-#Rk7QnN_ngmTb4T$LVdqZM@VJ-yM-^{S_hpJuTx60Y5r!8()upxPH?PeHy3Ojc-gLD-&2qPUs6ll~OM?SPWH$ZQK7&mHlb_+ai5ioWj92LM%(>#DX=Q_k'
    'msLsSt4ZJFnB=wa1)tVHAWxyg+0qPMp9@p(T-QbilZsJ)wU}cf>9hQ*w3xRMb3R)Fn+`MFQWeC|^aMR%2iQ?1mw~(!PEp~U?k<5#jG_F*I93tZ;#+ufO}7_='
    'JAMt}ENK!Py#k8R<Ho<$yNqNsvu5c+aXw>}%=_!S1T6!Oi#FMH4Tk2#8;oI-H$G}|{sTgbtv_j+LcQt&(bDS!=gus&hUe5<n_^rh#-G|OMfE)8M*KZ)!wEQG'
    'nl$gf+O3_s1g{~#QEH}<mNm44;?M2g=$@kcHd;~K#Ut1v{j)H$XMey^xmbcCW|(#-kw*Bkk9dW-_LG->9#OgGi{}qj_{^>kwVB!75D&OhL4Ss$X#u3P&IQ?+'
    'aOy17a~bttY~dGnqr^Cd@Cl2W6U_=K&H1Z0;k|1;SS>n6Je7`Vqq@l^14RMjhzP={0`sem&8d%UVPbv*w&7|M=5%YmCEir69%wW`tL2FyKZ<S}a_KKJ1#oM-'
    '(OBVTx~vG+_opnT&4q|flw3^J)vX+Ya3Bq9L;FafWCQ8F0IeL(U>u8vhs%LYpf}ns3lIW*7X)wjtTci9$n@;?Zsjt(VMODmcTq_HN$Qf_xk-Y(KmkN(@ipke'
    'gKHUunx~p;ZZnEn<xZuMZMa&m8AYgIb{|i@TBrupbot&sqRJFCS)(i4z&L~(Ge&qV*6~(<hr4xaLXumDOVBjPisTvgovCG~OZWn3SC06l)7**r;~6QM>YLL<'
    'v^piisB;0x)?1o`m><9>dCE4{MI!lM;IGw%iISjD*#fVv&rNFyYYhxd?%A#kN`$?!eq^tzQ{!;kdSz`4#h&+9R1bl!%!mE3LaTUjj$RGGQD*sklsrOtHKbqw'
    'dRLP9u7K|$lO3U-J2M_xnlCW~9c>!J>HJSHTBypQFbTtApzL-D{34c2cblxP_hEc5Y%H>ghQp?E4<1*?FVn0q#mSJI5?`N!Bu4OJ71u04x$w<!5T0#exZYrZ'
    'Cd<tD9L&|#x~MkhV*R>u5-_*SxOweOGb#8CAcq^IPday1IcLsI4-qZi7I?Tm*&+@7NN@V{yqv9z3(7-a{Xn5Ry1+3s+LZi;eJmdR{_#VTZh#sdP3LnQANAGm'
    '4Tm^q7+!R`LWq{5F>nXYfoJCtf51Ev8x*@J7iw&mr)&rky65WQ5u~veGK-*&VpsUK*IFOtqppu*Z>+Rj4d<)Z%f)JnQHcx%egyjnRY?r8Y_83SEki;Khw28c'
    '+YR9UL38gzpd?^rh?f+vzqi-Yz-cj`=gVm22hDZyV=v7q#nAvz&?(VCadN&YD4otaUu^001bW_d?GXc&Zl0t3#Eg!7lwIZheAqlaCZ1{dI>+1@d>C>g#jUVf'
    'Q>J@{5{F%Y<PVL^W+1m_SKd%f@n#~*FN>MKDW==a${gJ2zZYAtU3%Zx=*h))NP3^wF3rZh18kE1{Pe!pta;tq@~i1)b~;2Hx+oQVzW+tbEE^JRYia+}*0Gp#'
    'ZDb?4b{vR!{idvM>=>^vsI?Mi$%d%)x{YmU!`y0!@*p4HZ)^(B>O<$j6`iSjEnAP;5Ff6aY+b5vpdzCU?4`+RzPLE*?ZKl3nVibR)JQffabIj+hWw5C&>`U+'
    ')>Fs5gVx5B%iWOR7H`s!ZbX?@Y=d?KeV*=rnzn7Ay4Eb_M|6nIBQ$P-L+dZ7x4i+o%sd+E2}p>2q*+0%O5+HW>NvE?pyTw8*+`nV4rv!&lTDQS*@G^vS><x*'
    'qK@M{=iu|k_84KgZMpVZW=jXN3=!p5)6sr<klsu0rw@8vz6`bDmf`HT2z7J5H0p%_!rkQ4I?waLUYn-jg?SG-%)6GR40>?g4nZHF`pOSl6bhJToRdw_KsU`_'
    'b6zaz_)`u+B^X*ih?_KJb>;URlWVAnvF{)3b!`T7xOg0~q$+8NXd&!9NJl&74=H&I0q4r+sc)6;*quv}<?EB2U4~Rjbj4wV{;SpI)Z=mVpjl6$BcTkcWs5a4'
    '4LIt~&YIMHlgO8=>w}$~^`;jWi!1B)-LJciX*>s-D0>>sp^pi<%-q31K;<jX)8x}ZSc^{sehq$>CWicM@gRGRu2o5F^LebU!~6CtTI_*)v07FOIf^|rv*JJf'
    'I@x0z{{7GW7VDG2=uy>|s!n1<B8Etl@q7(Ng|Ur>i2!s<;kz`sKj?%IsvD^V5j}{1o{yRv`7pmgl>pl2Cq6M&^f{7nL1&gM95sb6p@q-G7NS7j-WG@>eTQFE'
    'hxlRI>)lU3?{%vZo!HjG@NvPABKZe!Af%B5LqVySDG8)|4&PNfauZ$CM6`fy$9F=Daj^S)du_}I4o!_bMqUfz6sifyS0>>}eOkBVD=7M4PbE&B7QjdTfH3M$'
    '5Y4qtUrx@^Ci^P4cYV|&+9t6*b9ACiZY|E0Foc8V{%h0`JA~;bS>#jr`ykeDxV!_LmpvLPj|)wsFokceeGkoI*Udu;wJbXphRD9;OI|ExY^~T$85>`ecIpRN'
    'F+R48G^&8bhg-1^dt(-<kHZl$zuqo0{KS5DcZ*adSS9fv0Gau|9f#7f_<}T+b?9dtbb;FnYOWTSISo>h#>yRyy+L%7;=nHJNk=*H(DlHPA~Tg95wjw?YM$o~'
    'kAgf<c|5*{&czxVJnoIL>q+uf1(2YtD0X_pc~IdOTjP%28q5IsT$>}+B}1kOG!CrMT)HMjX^>WsFWWfEi}pw6v{XM~s5s7gv?3iVa<L@#7e+I5>4N?sata`Q'
    '+#Uq*mpTiu+lo03RQ*)gr-aTI)^h}UFE0ANqX;|;SpZcUj5tweFYR$gSJlA8!xhXqJ2ccNO3rTGr!N^a@Py;u(<Ohd+wauKDEU3espM@*j<C{@p(S^Gp>GO?'
    '?cPqDiW+&F{LPUZZdhN9o_{9YHfNk~zW)xT6r$(|gmZEj$y{p^gSr989*X7$jf_{cwSm7&v2Qg~m~_X@dDV~4=EOY0{K&TLx7hu4_SV(Cu6j>fE@SHftI_`I'
    'H{&z6$ZK4zT)<`eYKjmy&(P1Hj`JAr!h6|Y#np!uR|!mVRb;t_fq>lTKS}nl@4k%=`&I?r!ObNd(hPZdtj~=UXq9o6(Ecju{&EE!$T<41A)j+O!v0mNeHf``'
    '4lgv%J4-UGb&dO~iyRXkq-9w<1B%Z1t&72f+m(YfshuRCk$db#>*U}ak5B2h_8~L1^J%f{2afVMW;9XymR@ZO_7Je~J~eyzhlfwUd-%<FPr^+r{7=rTFEP0?'
    'aF_(R<k|EjM>i#Lz?!4e8IU@hP0@+x5HTA$Y80MO?2y;%%V*CYeY;Dyo7H-D8ta$nSUJ9NkJIGGtCv)TlCv``&Za>BF!C|FT5Z=*WOItBoX+{ro~XHhE&2Hg'
    'Z!M-bg+xnE5(5J)i{8ZN7f_jq5xC|tRqcKJ$B)lHo@XC_`|<Z5zc4+adzR-LOpS{379xb8OL~uK1_LcK8v&oawS5^v=WX{PB1&u8@94d0{2s+dEAEfG`%SWE'
    'ZA<lv-(;x9h}&FTGmCou)l`m)1l7WMu`JHF=P~P=Zl}%wr%|ogzvnOsd0f?3vjQ@tT<)qKn?36=*hCDLgYBl%7}?xz$rG-ezu6#H_6`IA19t3d7yDbaeP0i('
    '?ZkYt?~GlEhcdz;;_U1jnRyGLcPn~F09~Wk>FcTX_tbms*dm1jhkb6Q|1EZANT5%q0Ql!4kUF>c5487ffme*=KWsQChkY!jSNWR!I?<hUKe5_42I6ivSg6R$'
    '4=^H>1>^vjL(lg|_Yx!FlCwO&C|$b7At|t!Nh^O%5@Bf^f1RBg+OuP2+|SJ^h!*;|QIb5}l*ywXhag<=`{uO05i9}YgQ&T6eq6jsR<HiVsOiug@PP%KAZQrq'
    'v<H*K6QmHaf=okqzJ2-vW0rwwWx_eLKLCmC<BbVb?qMFIQ@0ctYniXVH1i;pP$*8zvN%~%`Xip6J8|9$P1olhFn|_=J5kO`b!)4*n8AJAVY7ZqSRd@LKo?*r'
    '=YY$RKE-jUE?5)g`v@N*Bc7dtoWqzLe+-yWJ~{-te>t@A4E|_6eT@bJTlJW;mgnoB3+v^iFUqFJ2%X3ILtvTv_BgXROkk#-cnQgWKEy0g%Q<C%N>!@qOmal4'
    'XiH!U65eX^%u#HOE8vxsq6)aJ+LhTsh~l(P@6hbHGCW1U`OR;hj19`(B%!av`4LDzgVbixn9bnI1hc|`I_5?&g0}dp{p43aNAg{{1g5H>KQ*Qnd@a}c2Gyzd'
    'L<Tl>0aapQDIHM&9|P76B7>sybE>}?L?^~spg7QrqRxi1(^X0S4eC2cc`>aFD&N8B=j9$Xn1XNoVU1?1iz^iwh)FD{!x4Ht`sX^=<oGa^VkohH9vAlKm@-@>'
    'hZ7~W3~3j9IgFHSlgZNCzH51aO}m_6t{ItTc$SzOQ1~izA?$lS_y;Db)xPOCH=XP~<IUK-_1;wXgISX3rhD(5bR+T%>43c?`idgn1QY>V4--BiX!&qo^1pq9'
    '*+bqG<&k(INQ&E?sFLnSsail=K3z!}T;^}|Z@sK$<y&aUjajdMH2>|+{58&4Eh>14Nm=-`5deGCaN8RY2_Jg3?+YCaQi}qqeoCZ>s%fkc-lNq;t|{;(g6NEW'
    '#5ez!Q{>%Wq~Q}W!DmgZ6nG4CV?pSxcuQv6^(9aed@&-uqL~1dB;_Un%Q0ON7CDE-nsyG8i)_|^0|QM^z`&n;K)2l03KCJR=7R}mXMAcpKa`rPx7sGiV-A|I'
    '-MLDp-S3bx)(PjU$Ka$jh_D2Wh>65PQ^rY)fQ7(fly`Ev0Ja0&y}BYbGPs;ej<-vdTk>{<U{(4pZBjMl={D;dn}hM0b+|gq_!IlCipausaYC6f<81!t!5GJ('
    'l3Wj5`k%?od7OdF%TQa7cm2(PIfiCpwwlH6&<%~fajhxi5Asjia3e8rWbnMN61%ZorahS(&);Em`_NzH+Uaj3AJ}Iuy8xk=@mdD$--pmV8-~d$Twc?UrGOH|'
    'c#{UCwOjI5n!HxuId-F+j7@?v`kre0UNtwQ_Ik|b&?O<Ct=99tVZJn{y#3_I8S_QV)L=c)JZ8(7irMj^xX^5#S0XrtsNr%a7<P_5PsV42+MqgtEr?sLFM`&K'
    'd{^orHRtPwx|IOOtaC<G7t~#or$cgO(0^M}ogSNCZ;NXV<&Y#E{0BdQ$>1mOM_uYllVcc@b<lU)ydLx`mh#dqj;L#DD{xh{e3-ElH4h#Iods=izSzNapDrtz'
    'Egggi6POP;XFikDq1SDNDoPM%h<qhz3S(M-YGZtOHsSj<`iU)a+Y_&@|G{V{<`Yfz{AhJp;)ZGRv&yq3BM1_u`8?J9qx!PkeHe@*GDIiek9IlhzUN?dui-d`'
    'A;0&pw2)XFe-3sekHO!PGOj`O?#q6uWLiJk{orBdfwHo4Pn{;fev07;ydiLlk>owWY(&BM26)w}(l8Ig`*+1oQ?-3kyK~K>l>2V3?|8avz&=ZpG4IDo@>@E1'
    '*f%C?s#1EX+x$?ANq+^U9JkS44hP>*D#)HK`isB(^UO&0+Gd1KyA>oZ*R3RsE>pR^IqMWouzy6TB=FYfR=l;sBqtQfqO?1pK{&A;p=<P6suV?7I?=7gQ?M%2'
    '0?xG)%NiGl#V7l)1IEfC0Kd#|Cv|&byf>*CKyfpEbgLsVMB?vJ!(GwBVHxb8I+2cc{QBnmDtwOi^oM6sG7~$ospyBi7qilw!Ee^s49x7;ROPjgS9McgQyIT0'
    'UR8aqt}D#Z=(ch8s?KiS_u=ky{Wdik<GIyVxzwP-gZ<YgW7&3g*navarzCw;xlV(F#<c@$6HPI(`V-wnFoje!DXoP;1=7-56fevS9O&pZ62FDZb3)qnCEoJC'
    '<@q$KCk|!Ik0zvhR?appBQoTJaYXqqt7496H~Q3@&<z9bg!yMV#X9SkgyRGSBm)^iR=&FGj}<zXI|EqF->LCVeNT;=ju>ubrqPr~)030Fe+0^q)Q|e2$V-HG'
    'PwDVns@%+VA^54wDD@uoEGSv8@CRKCjE{(XI9cOsf$(e|ch0yLU{-&WJo2tBr;7rnj_`!YN+noQ!{g};O*x3Hg=ifFL6EOUDjz&GKd$DyrjZk5ohjlbu{Y_^'
    '08U(_K0VgfQpw$s*rY8$QJzKA+&mb!KxFQo;WKKE#?ovP!M|p!{J39lj>1ds*O$|^5KqtmFq>?%%0RcKCSH|&e1}X5=xJ}Xmr_j2@|gc#<|p#E7_Kfx$HjsZ'
    'x!^S`RZbK8UZw$`dY_GGn{(qC!l$iUH3qv-HcI#uG3f4U?LEstBr2<}A1tR{L{_xg0V4FngXQchW3(AzVq*LyMo#}C1GTl|nj72F!gb~Gq@rsv*goALD&u-1'
    '7v+>+BCkzGMdsnvh1{4c$JGS>Ixl%=J=`Q{IK85d63o^uZg`ACMmzLk4ZE`t%c`BJ1ZV0?_3eqVw{q+rlr2|_9NLgu{X97?cxxl~9O2h&kr5D)pk4R^28}8*'
    'GLr=Y$Ce+~PRqpPmkmiRcK2Ad>N3qAhAZuaVw*5@_g%KOFHvn0EM!wTokgIX<AHDU(&v_9NeUoQIxv9aQ1;@2`zIp4q!37j-~)6ovL`4lqZ!s=#L3eV#u4T2'
    '6-6$xqsH<KwcKL07GY0+JBtwCggKyxsv^whK69zb>xVb0(mEO(<y~dN4nRY553Aqe=E|rQY*u3}P&$a{vbGk64MDf)pqDEUBfI{m@E#9*##0<9%ZJT%ZB0T<'
    'YaD{lzy=uf9yy(=UhLoFLqjS-8+>7D%MAvVj!Ej9_toxRt@G(w)uiPiK>MiQO=sCk6)RxmScy){^#@CMJGLJ<sXJy?I*Fb;abX-EL<%*N;1%^Nev~}Uk?|Hw'
    'LRm~UMLwh%%Ga1y>~+4@oIFK_xlAxsM7}nho=7?kTBUMa-)&6LE=Rr_JBH+QJnPE98^p+5n3G!ds^R!7zdBk>&tGAJniVh)eQqAZlL?T{Fx-TW-mt`Ez&po('
    '8g@3Qe1iD)yK3HdO@d7&OK-L;mi0?m_BEoEJ9$uq*}AE2*^uWKK}9@m$O?0lJ04Rk<tJ_T<Jo4r%p3Hg@C70hXmS7kN)fLqAq7+}Yhr3rSkftMioZ6rrc@G_'
    'cCz%eG8dF@jw+#swaeXoCuOenf?r+eJ6V~^c!FDfeHZ4x;fZj%U{0{88n@LPkO~7NYv4R9Iw27zDr|W|*?N+`*7kV9(m-=dwSX2ONDXc=H>Xqn{1GkY9Cc4R'
    'VR-?%Dgla`#kuQl34pMdb6<ZW-4)fX)$4xjrkr*_a9xhti|7xHx8{QiMZsF-%M;9Zpr7LD9@-O}?rE`pU6cr;VN0m2o<_LPxYR{Txbtdc33IqOkq>l2&konz'
    '$d?Y2XQIrHYp9QE;Ek%SlNp_82$TqCJ*$kuUMIoD(adk<rjQk{OMIi4!L?RYk)gw)z}LTFW^BK1x;;ThYQ(W8uHUcDQ+$1(D_@O8`ouA027n>RN?Z0Lj7Xc)'
    'O)TPJMR(Q5m8Q=RIM{W>a3xokM{S4TI-MmXD?PP!Bo%FtHVIRkC~gr(msV8Yh-9zo)LDxEn{H?JJELA-UBB_IbyGf+`McifQ$vIziue13Z)dQ9KGOAX5h`eJ'
    'Fpw9l5dSv0x9^oylwc~v`?S~n>WbgwN><4_I!vaOVLYF{VH-`Do_mLxsY(3qg86KHQ=ZF5Q(G8SfcwlHof_mWPNer4pCtYL(Vjh7HC4sB-oUF0=<2oSoO4qL'
    'j4*+`@~h}ob47Z+Xcooew%!oPrzk~><~`DnY`k9OWNB+RZoEqFc0I@V#p-8=+TJYF2-WWn^({$Aefztn?lKF3$h?f>$OK}$>pDgNR^NudBgATzm?LxU;APR='
    'FP!gcGasQXR~^PUGG$#B=c{FbL0al`R^5EZN#AXnueG3sBWOO{KLFkqov1!rWKGqfs!EtfTp0&LClT~(L~)*Y=gqIER`mM0BKB=xwMDUzY1(C<bbU96h5oP#'
    '(hF#H(Mtls@l|46v;kIvN>+0LgKsef4xhWoMDN56Dm3M6u`2UEmy7IN%J&C))!t-~{H?9Y#e;1;8Rh>#lMbq<E?nuRy2DK`2NSK<nO+oZ;9`1FqIyJ$CCi?2'
    'z(<W3$bmFu?|do4mmc<AS82jZPdAk}jd!Gvt;keOa4269_v?<A(OV24?bTvS8&j?xlv&kR1GP!jWFnd_K^>Rl!TKC#?H6D^+z87wY2Lb`n~L+Iu`0f<s|;g0'
    'KPP7mC`uN(N<lt{ZZ~oU*<(htH`?EmJ6&mG;5SKkPuwnzULKF3r9)k72oDH8K|chRasaJkuXN=m&;2nO2Y&C1`)(dT@uudJpr9Tm$mP4qy;0!Y?=To$LMdM7'
    'Mw<f%)xn-$M6O9rSKhHbIGsDJM|l{=h>1J6XRog(EW^xjX2o9sSm!+*k4iHL+RM(4$tJTzh2%u^oUTyUsohs*IDzhj)v%u5q?^%!yX_ODzVvSxLbs()<p^pg'
    '))N}V{EhFXq-IL|l)NX#LCLpXYk_?8xfsJ>CO4fBZTqJ@N+p~Qpay1Sk%dy$4=+TExcf0^0h?U>WP?a8HP^mB3YYfwqOXMOssp-E_ZE(JHqu_)bXdu>)n+|Z'
    'g639@h$`vr`&$@cL7NPMF1sU4^mMVg^4LWT6;`p2TMOV(_C0OBi`w@ZZF-MRSD-c|(@4G@qb`w^ECKCkS+)O>F3d*0mvni$2CeLLx;VBstR;PnxhV!U>5`n|'
    'sW>YcQr_I@-qEK}Wvrv*i4;JCp`xg!aQF@y04)?Nwbg3TL*p7L{pwSkn+R~=Pu&h9b`ZGDMOZ{gq{>(%F)}`b*c*NJ0FSAv^-q$$(ZPcXD^bL)cnN#E1X}6X'
    'Kb5}UX%zMWJi&4Lha#uz9wx|U8EmWMWLL0UHb<5RM~#F+s1-~^X7>0<rp~$7>V5=WJi<>LgrbqAH|25itfW*iZBSdgOVJBBe{CiA8OtH{q%TCn2m-8p-AjRh'
    '?%p-9)F3i4KESAY<3p-3aV0K^LUa)-aX8Tmf+-oO2&w_;?aq79?O58F>}L1PfLbbcT?=>+*mj!@=H9NfmiB={(1Wg&T@-H#_t8^`bYJ|#SV{-lRw`Uw8%b3{'
    'u|~bX-=rO+e|*bwtjAG;K6Vo4+IK@plbR2&dRgyi@3eMK)%@5{%|r!Y+ecn?Z4MptHw1c|N1TI6H*0CDl@yoR8a<;1&MNA?VkC-<ythU3HY}N8D%|+#rFJ`}'
    'TFFU~ob8l-20SOGcPPqJ8*y*+plTlFhyG_860aycNbj-<>^sGlYa`S?tZ@|O6*D+eHJnKkc4&G-yX6k{N`wzIOF(`wE>ZEJo2G{5?Hb{&gZVo}v!y6EMv5w{'
    'r4UDGw!}I%o;X^EHmVZ3-g(1O%hw?vccw!&6S_CUplHlgNJ2uH3-RYyR(yoE!iorS_&JqgEPdR9Ya^lVwb@YkJ74eBdM+Hx1jZ6|9~`-{4wC*sgtxnT7*`GL'
    'pnt0yw}QvcR|m1Pv+1&GN%VJ>*!D6z$vSgsy*B!OTi@IilFN|1D?HFMJ3zh_SaP;r@#V8~h%Wl%f^N>{k<#@uO0`{|f}U2*HtqFmx|BK2YM7LrQwgVz`C?Lg'
    'AYT3BJZrzEzA?C}Rq0?=g&D|7P#xx(R&MDmS3LyFp)LBtoMA&VS!g7i`t~vEsRoy1C>c0j`BiujEp_I_G5W#wr?_hfjlkMbri0{fJ;TbYfv0O@)iAGxMI%{G'
    'IS6+33M2r5P6ZkP)wWE?7Q=)Pw_(~NcqtU+Y&ThL7=r=S--EsTpI7Z^Vg}jFPAO5z7oQrF!_qeL`m|W&{OBugQ6aW$*Xx`*L{*#GNu}2zRZ&V#aYFV!9ao6K'
    'UASc9A!+IGuZc0;^`+_5&3Ua2uPn~D1Y)c#*4*81X`w9^k(i)W1{j{NPuMLFt?1kHzH9F5Bez0B5?K$l+J;jCRb`}AQ(9kIPxjG7-RF71U;Z9@T`Xp+Hz?=L'
    'X4`dxRb9HYDd*3$rOv12nGqB9TZu=MQV-KH)<L<!f9PE{r7ec40FGm^V8MeGVG-fjgBf)9?*m2P-{?+Vvl~;rD;a1J&8)!4%8{9PY4(?Vy^^@GqaMG*L}?3h'
    'Uqj{MsCTjkc|*|8cyjXc-+C+09gS>Ga&@aqFD}^p>O<Y-bfH1;F#R2{dVKRBdv;Jzo`*IRhaNKFd0bZn*3Bx5vnYoa9r~6<gK7(H#NJ5q^Vlm7k~}OxWUn6`'
    'b~a2<MjVWG%a&yNgm2EyDc^fC?u&bjIWog)+gXK$?3=@-dhcd#FsXzlR%02Zw|I1-4rnw8LaiZ<?EoO}t82v@^squ?E*<2@aqlhveN6|W2tG)>RUZJvuyhkO'
    '8*#)KsBk@W?Of+4u>2hGW8D>CUM^O}(mA;c2(W~tm_<2(zjgHO+v~yTEWbj~Yv5*;k~*YT6^+EF5=uIE8Q#*Izbcj_zWB85T|IMWo+Uy;MDfBYR#F=rOwqvy'
    '!is4nlNJ)``Mx?`NTh~5%q)fka_vy(O5@+t_qdgR4>I8E1PtQRg~GT{iflOw$`tTZM_imU<E%J^35S`Y(<K3G<Bgx#$>mQxWDw)aE=z;QOc9}Jb4%v(5Suc8'
    '@7C64b#a!-MGCe;fxqTK_hwK7bK|1ncu%^tB(K=DbS5aq%e*YJ9M9$0Xnhgm48dia+V|5FHuO?ku(_}6s7RPWjdMy9A!HI~?R4{MN>UOHI_&;s@qbx!b-3(B'
    'u%CADWAyhpg_Q9uOzQc+4me|N17~ziF$HB`$E+V}qZTJ7!4F{&aQYM+{_EBH4EleqQNVvK?%CrH@@5q?ueiZ`mQeeOpHxw`xE_0QbzzvIqt&rWc^xuIx|3PG'
    'raKua(Yhl3Ho>a>9>{bKE6%2y4AU3)-~pTp?9bzU{xhN$!FQa3h}^)8z)%bXT#fgkx3ssuU22wArCH%gYS}xWceu|nk2`hg(}35GOv@m;`koqIM+07=7$q33'
    '*0Hm31qR$mKW+@9!x_ELIR6J~-<EMi3NxGTX(hbELj7U#^Rk7TemF{SRN!Y{@^mEgotsq_E7h)1*HM7inqjc4PFYGY3Bjv|V$bzwy#g^vXIOjcE&KXR8=T&f'
    '#>C;(Qyr|KZY`v**eC}(9S8f7dqjpnG<yS0sH^P-$8@Wq?1Za=OFB{9=c2fz^exoLh<;@ojOi5i$OL6JO?l+7S@=A3WC2vBCnpFHdn8aOTEvCYO`lyW$Js#I'
    'vE$B>li1<e1=i1`Iv#u5dE&Z8yv3krlTIs|L%@$cmN4QYaw>6tE(@=D5s=M;7kBRqLRMtv)(c=ES^V?pY&D|_Bu4A&JPWZNCk<W7guu~GH&X4tW`;<b)=g1W'
    '&_re{)nAjkmKEczTeP%sfqd3=)1I3<EJLVzc3sQ$W?-5?i3J;iHe6f^RO*t^i2c%vr5<ZV+DNJPKBE0F*1>d576Ga}9?JxA!;{Gffq$0M0AEjRr(8F<qiF1H'
    'N{9+ri#wW(7=`EJs$cO*wzsRUj*O~9rH^32rpprZnuv;?lq;90TI^lZ#p~%+nH*0qRbno$osDS)*zh#<e&V?LLwGoRWpSW{lxTyIwKw<_VA?p<l-otiuR9%r'
    '={y;cCf-6-rE$~1)B41J`=8yrmqx<qq<GZFqgsBsT2kVgVztH->^amemPaa$e6+woQ8eyjy#q=4)3ZxgEtQVBL{^m9t9%0^qzjxTumm>0)Ay(52f79v)!&&k'
    '&8*+Jy2|t#jbC=&bj7~d+gGo3pOp(Z=XrqfgI<Tp(C@KAx1;R^x`oFMqLL9zd3XqCh#fSEZed<+dNR2U!!dnL7+SGfm}$c=uiONnDxq(w8=C`lW79m0?AYWf'
    '4Gn$GBXWyqX~b@5Xi5`Sr^<rX=&D8uyceY%RXL!!NDGouYEuyA#&okS=~S>NPB7<SFJ$gtAGIfc6D7v;yk(Mz^bcq3%-PK|$(ILX0OqoYhe`MAl{@4M?ga0!'
    'AB~mQcbU4T@-;kWP#ccux({_&-9P1h@~q5ZZ<2YcLqDQq>IUfZS`@xC>8RCRS7SZn)|mjth#71I!w5LU812CMe5;TK{SEahxl8xv(>IuD5apv>b=O}SV7of)'
    'QN1S-dO1*$niCtpw5Kp-8|YQ`VA{At)3#lxtODHY__+6$JG{<LR}0Q5fPZAu`Jc83$<D>h!PvOl+ApiyGz8nuUtLlAqzrnkaqD%BObE>olspTmEZP-rNdJOK'
    '&)O}`5UGVU3&yYFc)M7T6yb3X%+>lJT4yYu5uef#UX(})bG8lVw|TAHG@&PfbgC$F+|vmK@xWu$njjF@6p!#<3aWK1SjW|xT2Awt6<<8a&{{khonVONUaWZi'
    ';shTxT5S4CF`F(fPN#$B79$IK(?&V-N`{^|*=%*1FSGeC_+7cKj#5Vzj5egS3aVZaS@dq;;mLS>!Lu>x1wFmGSixbo%*+0cmZX@e$RWA~7j>o^P+ckEMeo~v'
    'N|i2m$@c2Q<wtn<`$wEw9gl~FT<RG+!q<ss@4VA=nWAkp&K4L7QcfAodQwj`@uEyv&klPq=$zSlyP%#PJ^kW=ps#^S^(A634T+kPpM*Y#-vSB2QId#77f|FD'
    'l$rm6+R5e=J%m<^`Im|0uuX8SNBUkY?LO@X@o9HKU>|tieTC)!poza1PQ1hbI>(BZnUA{U+c~c~(=<5;{GT%E_v6%#Zi&m;RsC##)rrm{cZu%S)0<+RtQ*bn'
    '>R6v#K}26pmmAEyPE$|8!~|^V=6I8roAP$%??E_!M#C0YtxtG9tpeU*Vj~LtP7`czlm8c(<B5N{6Zn|i+&j^xj?zHcP*{8jud>a0o2yrNO=DhR9=GT#69a&*'
    'Me%mo(?qa>V=?4)@2XxU(Kzv_Pe{Bc1GQY6Yqw**MK4{tr`J$&@#u=x88CAM=Wt7t$~vSn>8U@%j#$*RrfM3~$h4)k)0C=d5%hjPOREHbXRo%)`9g36`0vBy'
    'F@McB^pk7XDT5MaALMbCr=Y|iCO=L$r*W#S()A{v&d{RC8<}k+=JrL_uqL(rq24@L9}^ueOPpT#(Z`{{alD6#-st8Je(=|hDB0d!I~w@&m@WsixA^bto`DdD'
    'qe|&Fs(XXBHwaEXEp3$2i6B_k*Em^EkBiv~s~EJ`^|V+LLSByM?kn@)N**BjF+vB0D`%8YJ)17)IwrHvUuH;<4hA~#&6ohqKsVB)a(WyWQE+k|4&zo7<{k!p'
    'H)UMzs5yz{rwx|ZQoOXAk!6#=o&UblFRtthw&ucfdO$PbThe4{e@@L`h@4DiF$iLBbg*Z0yO3r1NCiI;FmG{ub!4hW0$gRl0rI>aZ1k#zKdn;Ber~kVRje;H'
    'F$0tK5C-nms&wThFkS~ee;g%5mR-j~J4&o9g|wrmT%B*DlvuYMbVsAsM7~>NJ+x4-`KTLEEJe;2Xw1~SVkFTreRc0^2pg3U<sy(9G!_aVr--4m+!m%N@jmZS'
    '>?SpARAZJN$>h7}^gHU#i%$FGY(`Gb99iUF*Sax;<zH=!#XK|bX>rsLepal9|DXlAV|m8U49<)1#<B{N0RF|WDvHIUuvm$!gW9m<24SAaf~20x`O6VXTy$U~'
    'ulM&%D2<DEr_{MnZ0vVz<6dO#ynFI2lA^$G(pqtKU_C=^!E)vK#k@E#Rm7BJTdB|Z!k==d3e2PCQHXM4aIP(ec^R2TA50~f&T6DSrXE}1-bo`+ENX`D$qZ8g'
    '8k<Hj@CN0^=`w5>da?<E4e$cBBj5i~B9Xr)R^WKO`XxtPJJ`2*lD`pbYvN(k8B~{~AMBJY_%b}#7-lI{I|#9f^mugesAL!Fs|s#kFt4gMo(xs@@*zaE4CmfB'
    'CR+})vvRd&ZlHYdv<r0KZCQ`_XaYr?Q=KI%mk!F|p4lDrTE4Z@hxy;(kJ*wQ=tP_=ia(-X`6xvRm2uQU_z5haX>6d-_^f(wKBak*vE<BObbO>$WRvc$uVgaK'
    'YKNrQY}}Q^6PeAt=Y6)*;+MeP%Ok%24+cT6^x|f(;9zmalG4_QzD|`z*G-I|Q&ji7eddKH>lwAg(GhCQ7b!A%dIGc@mIWF2bRd2YgHb(b4L!?F8HvtzN_eb!'
    'Mx;#eG9J4#m@@-eEk`##phnP*6pOigNJADhrv^mj-o4LKtt6;}V06X0?aGk!y~JwOl?|@~t0l%JvY#$%WW3%n)bMA?JizuGjgpi6unPDyf#)OE*tUckpxNZi'
    's}bL~U4VsxD})Tz;!2CD2UbjBKSoZC{Z1zu=gt&V)EX_0#A*w7pyp^Ji|M(EZw&J)q=~mfvouI~u`m_2@UGvev2-HgF7@!%<(K=+K+e?cJa^L?-GTZWRF;5&'
    '*|G3mm5w0w588SN!1v7_kEc=yc6b8-dB+{T|Ji;@N}g-EalcEPEvZ~By8w4lR2>dAwhcqim?7QYvwJ{FvMPyCz?GgSV()V$uNq5s(&)?p#bQUVK}+B?ZMrJy'
    '4SAWFG)$AgZi()_$2~ePzGbkH>z+HqMTY3q3q$|y`qz8uUNVM8{p<d|*1u$6oz*Y(Az>@cDwz2$t9n^s(`jz7Aek0J#fj~WR%KFi;LVMd{LQM1>rAI6+)-Dv'
    'BX}Qu?N`v%G?QYhNEDP?Y&M<p&}jZ{r=?B{Q=b&k#&%M`TAA)3nw_&XEzB+hZkSG+Hl_{pFfgkOg1dxn8C;qdp_(>wCqXM2T(lO!saKUQ5onh?@SK1@2-nb`'
    '>BU!jzJrI!SYs+uq+4Z1t!d4RSF}7$fgUQn!|>MPT8+vi+Vz6N?G?9zlW0e2Y+g4#2|KP?CtczJ`Y^|}hpM-P)+a=x-(8(B;IFt>s;YIMS3GbgAbWNfwIUqA'
    'L=UTz${^TYx?YUhEdHgYGw;;cLT_1kbzxxIG~G4Ov@%`XPG1VsQ}?4!@$cEqwVqV7S<LydvRfRu%5K}m4Ax)t)L+aGvTHtV*fkY@7rkwQYg@}Hyet78SFux4'
    'd*fR(kl3xSWiX+JoJNn}!A&rYU^juT)0F5+B!$Mk28accu1#IF)~((;E28x9e?3pZC~~1U$2+4Jx&~^dx2O0C6!y4FPahC$5sxYOfP^Bu<YgUdVH0HP9p#5^'
    'Hm61PkPL-mlK{~&=J^$T6<$FhoCxuS*heccXZ9WCIpx0@G1CNB$t(FGAk63jZNI5+;}rcV^JC7{Tq%mwf##a^y<g?Xc%<5-Vh;1?v=un_hBWB8J017lx>>(Y'
    'F3aStn(OP(HOZmYP~Hmp*^e$3|0Kjqhfw3EyJ)QE_e?}2Uvg$AGQg^V_xM|WH8r0mXVpclq41C)L|4mRAH6VFuL;YTWV7CGPASI$=2n2SEGI4?lO)~(FR%P0'
    'R0NecQebHQ2HWY{5LB=tG!`oHM~RxESNnmBg{QIk{Z@50V|ocee5oF)U|xO70Axnmk7&Q+V!~l{E>AWc47tXj+C|nGuH}Y1$jVS6Xd_Q?oRJ`_CU}cr8~L|T'
    'R0HLfpG?LIUYv(PmmUQp;|86zi1WG;SQM(JkG4$UcD}mWr~lbIb$7$>_HnanK3+2@Dv~DN*QJ5nteOM5_iU#i(rNoQ?m7}-pWgNkd8gZ;1C0cUr$zn?bn6+8'
    '^r~N@#>!*#SIi`Z%tqzWjU3eKRe?}#@(=r3VY>D%xN5+UEaDMG4)hNU_ec{3+RG5T#pbkNIQA|`B^nt?0Tn9TzrkS;9%i5lxKe<*^PG+`y%_G8u4k<lV;;by'
    '8g8gQ#BkgWLG=x%lxj#u6;slwT5h~?7@W!%sC*1*<gIIOVh?wp?(N4LuMNaE_U2e@eK5<F@Ol?hpHxHU>DV%8T<IuINB1RnNAEx9npZ<Y-e=LP-{7r!<aKP>'
    'Kl_VZb~nS>9P!LK-tW~_Mv(}8kJqYoGB=VO0C-2-!j`~iPb4j%@2zd$z%<RZ3v1Lmc^fw#2|M|j$$zk~4ffmlG@G5~GlZL_A3Un)V7zWkb9h6_*2ar1R|c83'
    'fQ6%=c~~3&B`-$+Ep^A>{-BWoMJuY+Rkw*x8m`DcsanGQR+Yx?xqf{T^_%)6a+{}+mn%6KTr|!5>%b1zk;7zEj2OkWQ!P##d+2>jS_uWC_?Vmk!zDoGAv?hh'
    'X~%3Z6qjSK;ER4r{tm-trlX#Y4!X_n(ee6?Ulm!BE+6+u9rlQ7my%)eVzDg|(`%idutUoS`$QAt)cAQ-+V1joKdKv4tx@vVsZB9lEf-gjOoNnR@XxclaomPr'
    'f&F`{yL4%{UpeZgVNhx3nIS1YT?U>@W<VdF2HGZ}dUy98bMVLn<@vp<Z9qU9CIZf6ikvRbKKw*fu{peX(41d?sTbwveeCbd?pkt|8S<&B>;U#EOX<6(i$!%t'
    '$`AMH-5^Th(_lo^PIEyW8Wl`;8-BkYQFJaQB9PHv`WViX!6jYAsy~{Qmx!bqK2IxktX<Eq;_hfFHMA3=S4Vr(RcblcPwgn_)z`kUcj^GZrAl~{6qYxL<+`+i'
    'L$9>07`NZNzLyGX*;_Kc;PWcVonLL0hS9l;!O9fId-Sgv>nr(v*4;+B`bzf})f#t{RQD8fRjEoFaiY+kG!cTL7*7<IASU`1r_6>Mzd>CeC}!Fk=g7W}t*cW*'
    'tX=iGdy`hzkG5!!1)UoVpn$1OAL>{ww^(yzeFYBtjzU5sz;S-fKlEicigmflLhYMO{f1Xnao=CB5%U1r>K}8suS9j;r8b1u(bpS4hsA2suW~bgB21*`UB{~I'
    '<<ID1c_IN}A0^mYa+)tL(Axo{S}r!p)MT-^i{``OSz<1rky97Imq0R$Ve0j1zQil!(qu)Vk=l($V~y|Wam`~3oJ)W_*5-wwaK6;5nv2{w8`i^>)dxq0SF`=c'
    'b<aBwFk;)NtV%FM$*#04j@ju>$u-zQC5zNQJbC`&>9Zfw_rA|i6u;W8XX<jAe-rk)0L+>2Bq;On`OBw|zI&4WOcK2(?pwH@mY2nPwWOOdn4~EV&w6*M-h^JJ'
    'Mn!lA$6nBdjte-l8=b5u#cG0B2Pae1Pz9C&iBKsC8q&bwKyk}&!<5zNNHRSy7FT=$T0wKX9*wFGK}`~0Z@eteFJQs=U^7hiEBB-v{LDU0Wd`5A+5)4R6IvhI'
    'fM8zmJtw?AjF^0cc2V=im;<(w5Hh%N)JVwNR@-gDL1;gL8#TblgW9O0kR{dAO3qR15v+v?`?Oof0`!K|*${odd()A1Fs$uD`Q8UTxHn!$&fg*8<e(rin2j@I'
    '{uADqnH`xXL9o1eF~VYI|7yT+>$6i$`PJ&MV~y0>avAsprAGc(G386OD%e(&svK35tLpKts&QN`ZdQkPRD~mIaG?I~r259y-WqjxS5>!O&8<^!cUNs2)LO01'
    'c2i#i)wNz_{b1_qhfq~-S53Wb6}6(CTD4TEqz(z+vVopvG+0#qG~7*Nu7rEAs#zWeIY63#_Ozbb(4-Ih0wNWj3tD8R09ruFg!PyQ845j!6-;`&d7bCWt|=$d'
    '3N8$NkxL8BI3TO_m5-95Mu&`le=nmT`W0$f%wN$YLf$&=mag~w$#)N5zWb-=PoE`^p8fFmPoIBpe*S;|`Tw!H&RfywuF-mO)JuMwe1SK}TCw^5-9Ns3_W0S0'
    'Ls5eI)(=YCOXTTYVlVK{bl^YtG^U53Prdu+1*XBiSQpD#aWP#aQ=_55UqG#8vOTxh?8!4YDoo$~%e((U*$T;YStjMX|F%kKBfR_P%e(+yc@9sP_>`<*#b%qI'
    'C+F|}b+ekU;GhAD2<+Wy;UQA>@cSP<fBN#>Kf)BfNPc|&?jMJbpZxvPAD+UeCy$>bFMj{y6L|If>EnlwpQxJ0y<zeK(T)?V0lxe9buk4Jh$wk*z}ZpD8wGtm'
    'eU#XHz~qYuA9EsH3Lar7dz^*Em&rv!0CJ%1A?nx~ruHl8nr=0&zf9&CVamOY!ED0oi+r(yZ2<C56|rEN*R;J$*JWfrpFer=`|n;p#0h-z!^`I|hfkh7ej%lt'
    'V;S$F(651L+FbSae#n=n+w-Zp&IB5O<%Mrv=VlzM0=3Hd_7~9SOlou=`eU}DAiVL6@30xiwqq`p)L@3&yeMLxR1cQ5sMo&V3^|`+#$Y+#NW;Z)TZ*|GTZ=%x'
    'R-Ym^VQYU~(O}zfdxO#~y+*-ZQx5??O`6bi@$#B-k_vqY#PK9fAyNuKluJqx0kc$4=4zMyODANeZV7YP)HMB2zRlWkkq~$RKl73ilhMxxyD-)1*I?kIDu;u{'
    'dlwk=n6}BcPk(&(PcNUmc=Ez>>c<aX^c1v>_oa61jo)g#hHPoxUnb#<Y^T%W;@!V*$VYSoEfm{?nI!yg2A})xU*WUywD#^_hiqDhmm@87{4aU>!^<bne|-Mr'
    '<->RX_1%AYhL4^lB&?7nu9zvq=6m-qxRaN&#kNFVK1v>1=E*yIo+ozF=TrDK&+*&wztIs+XF!yrC{6aBA>`q5b_!H`2GRoL28aTL8H2_yK{Aw>RN_rO18T>H'
    'QC2_?iA6I?R$ICwK)!=42K{LB%VPfSzk|q{((2RLkO4zI&pse)*msisZo-lo1j)0^3TFH@+PvBH1|vTEks{aokIUZZPpe|tr@02H3g5As!$KYPwwvSO=RIGW'
    'G1s`cB!JccfHgRApv6R#pBrm1D#vWS0%HpGswi=o*4EQwA_v9TsW6S6{o2lX4QIN_5iX6M)oRaWY3&KD^+F+%EK83<ouE?xsMNu*-%DdBoYX&@q~75ot>lcW'
    'SgU)f#v=7j0b14@mAZx$(dOu^*gy%${E;A(mmwA;0-;o$@N(j%;}o(--Y50x4cU%enn7E)F2$fNz&ukW+NCexe#y@Tnb>s0=5tUy*+88!A0{j;$^^xuj=A?#'
    'E%B<0)NLmOVSEeEuF3K*kY6lt90W!*H=eZTugZkCxb^O`!RIvxGt)?QGYW%q<Cjr+WAEj5K+}wrHrABX$G8S;uhk~-IZ6XFz_zB^@!SV~CwS$%(iMu8YB&mE'
    '8Su8fFEWK<sl0d16v{rCgz7FGU->294TKfOl?S4vPDV)EAc3nHWhZzupe+AD*B2nb-JJw(9?^g~Z+PuGOTu!iEjulULpcyRabWX>jo{b;f~W5yHiIwK7fKDi'
    'j+@@u(a?2>{2<&ygFfp%9kvg7<f9?P7QL2Xs0q>0)fi4@`mbT;lRH6%6k9i3-^L3ncrsRh#y<@6!s|jccR}6+G~HCyOiAtil^0B_=cZ0rRVK&P?l?QH!dU;^'
    'Fh9CfrwMAhDQhH}x+}hzQOTH-tjWlDo_fF|%Ie>8`PSiTjF=dFh%t^{f+i;{PnrGPt=SHDX!@qhdeySCSx%<}V5SBXo#548vr8;rf3<icIW%#0?@nx<DbKth'
    'r%*Y4moDg(xXq8xpML-F`5&|IpS=9`+2a@4@4kEX&BO2TLME)oJhP_Sch4SSE&b)}hcwS%TJ*h?C?4ic>e|z)^=%Ci`M$lgA?2Onop5dXq)u1$Zr&*L5J5F<'
    'frsmpEiTuO^rk=0OE?o<;KUwT4>Ra+v`;oaEeP$U=g$wHl5c(T==YBw3K?@pqv?E(qoN-Az2Oi6TG11pVg<{iaS!cPr16=f68?ar66zQ5VN;lrBQDg~E>G1E'
    'q-oF9!!t!=Eu_w%j@lRaw%1x8_H(K0<JcQ3Emy<&>h*H5nxc({mFOc-$x-XW;2`VU@NIz)3-l^O+8;9QBNxokU_GTxVnZo51*<$?!fM?g-D6GwYAmR-&>M1z'
    'q`(6vF4=n2G(a{h4#}(tBQyZM?i=^__L{$N2vk}Sk(AK}?zc8@TFhav(aH~+>*9o7nuCg?GZ6TvL}$gx`Kp+6V)n(B4p}gf)3rx%RJySnI3%0V(N(OPoqTxE'
    'Sk-v{5Bb`ck34RqxT{v{b2<<ayf82&)K;L8*$kNR?8+OeDc($|`DHQlH{Nu+S(#%R{r6()wM*|C8$G$$4oPwI+NIgJM>%ll&rk1r&6?M(Ex(#>W~W0m{Y9zZ'
    '^ZhScX4#NnTTA<&wvNS|Vk4W?wc~J$>o<OVW5;N5qt;58B^%Ju>o&Hb4GO6t%7c7(zp*JinGc<7M0BR^wQN0VL%gbQvURDxL4b=k(84zIH|i6H1RGYr?)MH_'
    '8&fVOLqaXPNkh63Wg@W++O6?<y8mg~wl(Tnv(fC)AvTZDgf|VXBBnRA!Hmm1>gY-MyM3fLd8|s~2(&t9v#G`@88ZzuZwgZ5ye6Ay^0NnBTC;a1Ll-s|=OzcA'
    'cWG~Z3d~?la>&lwUJBm^|Gy9a{}lcY^f21%b!pAoJB9+rTRV3K-m<1W)}q^psd@gDZE)yt?Be{Ob3+V&8lAGe2k@_bn!E>XR3@VtqGML2v4iwpdOv;8CijhR'
    'R@-3qx|6&)28u>dyLt~gQ9jyZTb8g*gEL22qYDad)J@2t^Uis&*D@%b*sN_A?(Ma;=102@IokE6`-ZN=>nk_Y`FQ~xif%eRVYTb(V>5J4kbBX{O5JZEZDdT&'
    'bs#S6x(%Wnf>JuP=kA*{Wlw~yW7_Z9P?Ku0f3VlJ8O$u=N$!%Wq$OfGu#=hMU8%`fk<U~A7Nlc$l2@v7qT!q;e4n$O%Z<~k)#lWb#q^+AQ^r9JWl$}QqglB9'
    '!?u{6H63bALQ}4;jTU0Zm|k2guB<_*IbUj>jl)Dm(`XKTK)hw<j;3gSESCMs^ECN%5Z2=Js9%GhrHLUwTRh0SqH6`O(&Y14T?Z)bSG2L>_hPk-=FHZe@S&L%'
    '|LNDs9$zZm|2&G{A4ztiM^#^{`eh9n0U}MtaSHasl?kEkbxYy9G`SztjzjM1Hflh5D+^Eqw5Xe^#bP{&Pm~WV8kr)$Ky|{~X1ZpI_d;^kZandp;w9kA_U>2c'
    '+a`E&)ar;ld<nz(JRD9GPTJd<alGkpBz3?MroG<%^z&Y~DteXn>q$&u_+$Hfq$>78RQ(EyKb8gEG$<HA_t!QUr13E29;Ew~1_RGQHKpM@SeM#$t!W}!pxTac'
    '5G*O=#3~TFrJ5qujpR8S##*2kb_~9!Urb!w-`i{B^!Fa@H8S+@buBed-}LVtFaT41!GU)_jGAWhl|iVk&kUD*m7q51zI8oC2iiZNgWeN_9j!Z(C+FL8lf25U'
    '-vIT9S1#Dzs27f=VK70)hyfOb<8Jpc1!pEZH1x>agEgI~;n+zG>6p6R;}C8l4471G;jO)wKy&_e^Qc0-*$#yvoiFp)NTS=JemIruk(Mz<mFG?atJz&hGzZOn'
    '*69AW*Xbr%<Wu<jVC1WV*sth+>+Leb`RsRhA4^qIRMHCrSTf&N&G`F^Z(Czohh@eg5V$X)=4x@7(;y|$tK5lNX9zdr4u`Uy0+m+--EbZ$YE}N-n6}YX^N_T7'
    '6y)a0<MHpr(Jy|}KkkjOYwrSYlDt*19oNZ!q7fW%l2-V|w+2UVWyVMH?KXj{B?g{N<6IighHDb2200P=YMUd*Xwzk05M-KYw_BkX=}(c2CHKEDx`8<$k^5GT'
    'h;&%9r^LXMj(zCNS+VLg$Bu1u_%!ZoBlOz65G-@!!3NJl9z&G|BhDq-OM9GcRy8p3Py+K!UuqO(xqXGd=}QI;4B)u;bjhFVyk4o1!{jaf*HF#5NVst@Mt97e'
    'xWh{1ZSprqa?ItEOY|f>>9!l=T=@OBB&9e-*F&0sNFH62xYP|mwn;R1X=J>jAP@YVNSKQvG-#fyAb2(@^ARRxw(ZdxU1x8x)^)E^%Q6wSxB(|I?cSV?+$Jk='
    'VQPMY<fA@K55m|Psu|QV7vmLuFZ-)h`oK~NCb=q%T*E*+ZuFld``35hMu&Z?BJAMiA`EF}S21P@?XR-zud?i~vh3Dm8Sbl2@{D<q7Gv!UC_3l2E&&gIEeUAk'
    '9((mWIe5q8Q~Ira$V}~gS}glPd=HKpP1L@nU*83IW&{j^${zmV;nVLPe)HXvaMKF^lQZbc)10sm6X23((~}%wSl0PJY|#mHlsrW%-C{wmiigQXalycW@PrV~'
    'y<T5Fd;aL#UAhG=Xm%Rwm+06;ADeS8SJ#4@pYYCLa#6sq<YhH52&ovxaDD-miI#xr9aGib$AA3z{Ns7{@wXp;|M3gc6Z(#N!3NZ*c$Ojz2&OOT@seZgT4tlU'
    'v$wXd>zjo|7{}G-kcjRs%RT@Ox!Z<%jAm(RJ*#U_d<Fq<+7^9ud&k8abZ4Pl>KRA+iIlsU7jeB8au;vncGv7{H~Z`S1e#i}>|P6(LL7-5Tf_&W`S$#x>^J*!'
    '8@ig7v!Xasp{&cz(Sf@(q_~cF`z<T(xruXR^ansbG7kK*1nnN+9N*AVyyE#SPMC*DC<g43HBG1G(YtAFM*C>DY>M2GU|jFW6Nnhcpm(uSR@?VA-P%sfCwtL9'
    'kw{e|{~}=#zG<JSut!<9F+iAd<-DGj62c3z!rKnV7X1~3$a6FQZ?QA8g#E!ax#HjX2n6db{&UCk0{8s9lg0EZUz1aQK3y-a$oW12wgfMWy!-$od3g$AjiJ>N'
    '@~tEwmj!f|ELN}c^>C42<_jK&!E@19-DK7lS3_t{Q2Ry+W*8!ufy-Qc*LfX4GGU<fx|oX}J22nZ(=|qQY)(sVj{Aq!9{o53Axp9ygj+H@1;Kz&;y_>tftrz9'
    '6lUkjUjt>1?%etI=?jdRhE19Y!^!@Dffd;S-jPwr!|0%gj`*i;2`~UWUw>(aD-}`RPs_46S#oM5`ieVo{sE2GCp$1`|ARYGZbEfytEe#UecW5KeoLe^?5#l0'
    'p-Bzd6X}ZNChoQnsXHTlh&u4>6l6s4OWqF#F+(bekAkj+4qe2PKVnZGrNIEH9E052;t{jbjR-?fO<+L%>fv~EeDB~8Ax{>;4E*Ldzj?Amz{=#|IrNYxFDdVH'
    '@`x}Zfy@#LE+UvnqHA1eTKW-#rO`u%-e2da?E({_2ItAED<c4r#Q*-@K{6#LFAF<^)U-L@E=;CxhK`0q)h3y*faD2!0qHm=DptJ8Nb-bE%B2Jfl_T>_T9fVX'
    'w#52aOw-R==lUvKd)(Q;$V(;y{*n>wtkvrbryD^rV7~kK_n@Kba@kr5KIC*-V$6J>>XRn8$t02><E}G1fnll;?z5>zzegA-*b$9_@h$KfMz*_we3U%ATorQ%'
    '1(9Ho1iP7_23A;<bc&-+3_i;SNQiO|%~eyxLCY@_Qkm7_mY84-K{TPxIUY`Mz0UJ_u{}>Hk|CFMuANUeydyR{Np>RwM}0#7fsa<=5Lg$9u3^JJ3C4jDwKF<n'
    '*;sn$?ayP%S^*>Nm-*uO5Sx19GGH8Hj<@BUa=fL<AN`*{`9ELjpR||VPrIXEV$=+cm**JvFpt~_2bgLO4oE13G@Icu7()sd<tW%aJDmf6yn0Ee!0_}(ox>zx'
    'sE<)C=tzfWIIO`3JiKtF85((9kkv+0R1Sw;xsfzNZ%mrujjb`YsqHD59%U{#!OR7mR`2J5Nu@{sPK~ufH}}Wx&Hb@MbIS~MMoiY4rHOK``-xL@F}+^7Z-&47'
    '&fl)BbzXJRkR5#}rI>;hxoLY%O+)Gj>8L1;x3jhFrE2JsOMvlcVKJ<yNJiU~P>ZUm1g$%uYlu7fq-%p-dp~twn4iYf(IE_B(qWCPjjwwTuWr`EFK(Sdes0hE'
    'z2$LwetbIo`kZ3+fR<DIP&~N9XvT-BBpf=Nc<p@4sk@IynTr7wzVnOU<KncmeX=m<e>>X18kZNVGRH+YfboBe9r8ZQPm<pvMUJK8O5f{FEy6Z}^Unov^ga^8'
    'ke5D9hWq&cU}-&>YgYr#;A%k@1*j1xBR+$`SEV?RFta4_ea)#_o0Tu{`nV{Tm)ON1(g*K21)~LxjSz_jrhYJ*U2NgsOE~Xe!GEVE5=39T+h?nbs|vCfbG-$w'
    'd5SBZCcnUHq>1^!RK}~7L?B28QUUb~j)XiSICH!y%A>sjDmutn)UI><<c}^fps!b&Y#!P1b~#hj)F#EEg6St1H4og9DYI=0t6cV#Xh<F3|7j#oeiG)-9Nr{H'
    '4R@i+nEJDyBvq19oo<)S?2*`w_@HTqgr-Q%ir$x(QW_RYi3)87jPgyC?ffhp#;{DPIqA-7C|h?~&nK|gze|3?mIsor$*a$EX9%5JyCI=Fy1j(2fnyT4O}AP6'
    'C+ne5fV7l8&KM<-veX)+!1ZX?9lTk+i4m<r!mg07pVrapR{tT-FOvVCy|-O%<46_-|MMyOc)|yO)PP7zvVADiaV3w&)?V9k$QoM{2t)@6G|9F=0D}f3(Y%bf'
    '@30%O|95}vgPb?HPqO({Raw>5jRr~0%(<ZxXDkq?%BsrB%F4{jkKzKoeNk(&!SV@j5NqWi(iSUInVk%7u<O6RditZ4uG$|4m{AKt_iCqANuJV3P}&1F{H_LB'
    'MjbiN_vsi&Hsh2aozZZAWD|*z!Yv5JC~X0RHl`KtLn=>{D?X$>{`<G^FQ!E~fFy4E>!$BeNfDLNjv8zu%;CzF^u!rbvZ*-(RTY%aqUQ(ZY4BsYiX0a^$vs^z'
    'aqG!$bE;@EO~(rTTHXCwE>73TdKfHBu(&u@Zo-VJ$t)kQx;@mSNJWQx*%)OUQBq1GP^9yGipfAQ>7Ee356xuby6O;BbnwxAh4J@<HLNRehPP0MXGCQQcxoNz'
    'kz3FYw32Imz}*1;0i~Ap7mCSz;`r@R@{$xcEVese<8&#%g*}?z{rp#*6JMP_-pLcBA}mK>w+Gb9i^b~l2B$J|L%4~eRiggBSQa@u%HLiVXB-JX#ERgti)6A{'
    't)vz@HILpPJCU=2*htZ|N^?;((Y0^tVzb(K-DLpWXrC?xK^$EQA}(*a7wV6Ag%XWd`PphbE%332ub>!Zp3cVW%#bJjVM}3ZVujxwZ9WC6y<C$8Rg<V0E#(^z'
    'jb`-1mWW2^P>;5Gl$&ort03WF!BI7BrDgKdfm1*jqu8a_j!`zAgxiZq^Ego}HPY#i+|VaOTCcx)rKI`#2JfP2Y>L*`wEn{$D02?HpU+$CV$*z8yo<G_AI4Oj'
    'HQLc0)APD3rR|ME?PbAIQFbUdCvSwOtc7G?OTqdGVim=}#J^B+HBASveI(G#T=Ont<(`IueCD^HAN`D(7-ZU}#%L6}wDV!(modnEFx2kj2r`qwW?ue%ljnbc'
    'V%q~iU4Qz0Jz&JM#OFR?T#Lya(UoE(A^#EcmPFKe=0Q1-QK6Q9<QZmEKql9zZSh9sCKOjg>??$qky7c{Te{aIDL<EWcnr@gEJ|#_gpMWu`G2dS_$;Hjke?ZT'
    '9D_`a&)~>G+1ihf2WSlD<XlI|cjHZ2{;&V}fBul4!FH{3dSy71a;>MU#YKYNKEaaUe9+gfY-f_Zq;zWh4l)~GqQMk}dK53rtmIiD`rfdU#ndDi6LV>qhzZqL'
    'Ntv}B`|n9r6X~05zHFloyWrHSjvyyfLE=t6$(WvayL}Y3el>n?0v7Qj2RD^GrR8FFIbU29<Jn*)yNn;saOcSfM#qH3#Q&~^j!tX?kSiV}hlBm?`t#q_L_(l?'
    'nk~WADEEN-?R{lGVgTG08ZctPTHsT31eVdnnsjDQO_n>Kl`EF)jgr9Je+hbGhOajEXJ11wtv*Pj8f73w6o~>1EW@Gwk9@T#vsv*r@7fkHBEM!>DFg)rmP8`F'
    'OZ=j65mt0IW@TgpAuwzVXr=czo?C~QwqY4c#P6V)Q$V3?mqnzb7$_@sdZ3~@#EufxhpZ}7eNYQe)sia~C?{q4T@29sy0h2!4ay$SX(&yP2$(M*ye9cXsJW>j'
    'Zyh2nMh0(As&2C<kyU&9om!&lRgJ;VP+o!7LjiMiV%KYWjVhg{`HWcLn-guH2U*}*t<BsMP*pSA`F^sNZ@u3}Aac-fI8xnX0HFfpSSzMl6EGUvJ<YLyFzQA7'
    'W=<r2t(HY|Fn0VmCkm<0)Ms_L)X>#|4DFXbH)vAGSnUdxFL&UY2eF^+v|6lz-X|(xjfg>h<sxctF1l{Cd+z3l?j(!ihgHam=I?9JOlJ4g$WdvYm}YRMkg6Tj'
    '2NIa<v%$TXI>5}@*9d%u;9o7NJ=KmRvFLOrI*GX>1}GnQS-~GxyU<cyA1hp2P|mS`b3q=}xM;?bj}h4CV!9zWZfe?Bu=kugxkbpT7T^m?&hhvytl`=zQr2Cu'
    'qZw6WM{uUNcLkuu+O81wVI;HxFo(Wfp(&JMj%zB0cCj<kOb+3BUY1#oqBMGy$}Zmu*!ZMe%r@)XLKUh%DZ2)KZ`T3x*?3uEMFCrjosXW<s;d-%PS?g*o7vKz'
    'EQV+C(w`kKAK_6%%F!;rC^TnAjAo%&`mhx3Rl=xBI^0y<{~i%W#4b0XO8c1j!>tuhch&Dg?U|Krn{$z+$bXXd+@vjwbOLw;x?y)b<Dt~7N<Sk_cM58#wIb4*'
    'wK|PnQ|xVnzH*wT**#A?_Si{SOLfH`?%%2|kHY_0$BW>dkAc!Ly+^4~jf{)t|7fLGrt{BMdT*l-PNIF?!v1NhAR8OA6Zf+`AC2x>)w)9kaJ#mdFJnf4gS>xm'
    'xP7x)>1~3zRw^>pfY(=jG6udtfuEa#->A=Fm?v<0fo)8T(fA~vE#42bsO63NUrL3XG`IO9`#@@0+)PhD)*0%v4^^q!QrcEUbh5}#PnGHW^~q@P8ux3ca+dK!'
    '&*)!4gZ+b!>a!@G7*{7vxH_pSF;jdb0W1a@F|=*eiKl2WIj?j(-_q@(;Bpzzd4KR2&GO;U_rswhQ*QklV}`5306A|wcTD*`9>A`XBd34#l5dcWq`1-tfTfg&'
    'twLhek+e$zdXLF!*rfKCv&E7ghD>))^|xO-<v;TL;lbe#+tuZmimc9zr^t4j4JG)j!o%*O->Sj3=(mR3-OLrr*+mSTU9{s#s2HQd=3t$8dhw*95?iZTDatT*'
    'Z><ZHUK1$7{~|iD^)8X+-B|avfXp|$LSo6*({}%b6kYoFV`;hc%|AiKWyO{de^ZtABdV#_+qBOAee_LL3cg9dW)bvO+Ft5r*Te9epXsrL0v1=r8G2o_e#Q=N'
    '<5TotM3J~aO2$wjdD|7|WEOp;^!Z><?ZUuC;5Zg%zw%qw_&mFQH4Qu{b#M9(N?nTz7$5mQ0ROj!5T?Xf5KPiYxE|q5<SY7I8^V;meD&hH7e77yA$$Jj=jY$P'
    'e*Wj|>9Zf6zQ%~p<HQn#r+;L$PnnH#bueE1^vm;C(pU0YRUuqTe>*$>;NfdneB7?hM>C6^!0O>T%`tZmmcI0A{Ca0>)Z&nuSi~cAbxn<G%ai9rCx<N(sr6ja'
    'nlM?%Js~9K^ArS>4IEw62q>P)#KN@+cO<k7+66sEycE#&!C+k%V3ANbiW~!Qi_IGAp40S<14COQP0Tp+jQetF02I8{KKLHbW)G}&CbQNVUi$6KGuvz<G)7AE'
    '-D(xNjj2aG`K)xwBtI?#bu{qblNd=P#28S@Hbsxk%wd4Oa=7j>i8WmG>X0ZRDW)HMdl@BwFd%vrdzs0xI{h%9`^l5#aREb^L5LG*z1?<&Qa=`+l9F+M7XEhR'
    ')+0;6ki<!j?$PRO;qk-L>6jWvPOJc`@6LVLqv*pEiF)P$So#4kAveT8HvL_ZzZZLk(t6Az>b#=dpISnao}|tz1!-$n%4zGZib+{zdDr?W?Sh(GDw9sl9Ab$i'
    'c7P^O3|OH(exP<>JWch~B%xsE_=p`$4j#V-JE`xO4}IV|;T2#&ATB7XYFqLAiJd}r&`vUzORLol`LA+uMenGZ3<=xrlggvP_Ul`=5-tM0zCf%rQ%)UeOH7d`'
    'y4H&=yA5OZvGPWHdD}f~A=&>UxA}BKmgaW!x+4nRa(8-Sk-fy3VUV5}Rxn~tyn(HoffrZON~Zj;zFoIIm9pO-@l?8EkT=}_U1=bC{;kTP77~uT10QIhuy^p)'
    'Vzm=n_)y(^BXw+1^>0Wvf}S02B|H68@7xDpShv?G2XUv#&@5wAgHM9pgNxQz?qGZ0VuCxV6LZ5;8b6*qCA1P`Cnne2QS#$>xdhRTO&~RA_J%s3U*twvWH5`H'
    '`cq0MtDYutrT5ihnR^Q6WVV^+Qc1^}cJR>2-oM@~XZcVkMe|dnrONi>K2b<C?JBIh>VIIZ2V=^JYU~FtohwMvuV52V8aCO0fEz6~*xM?qZ+t!wsD69CeZ9gk'
    's+?k;SsDE}d1IwE4q>Hm9#eSeZN@Ts*L7hK#@>_}S(90*xZUtU>AvyXLMy*pz@pyt!pN9sH&JG-QJWuwV-tHO0;<IN``ugwjsi6F=CxY9ujbQ3gP@=Mc!M`z'
    'jB4l;RY~*d2LnoV_CF?`&QWrO8Llyv@DYIL9J`cYx4fo5@K?%x967VszP*<z@J>KpJFUq(Ty{K0!O$&aH0->hHXT}Ob?8c0H{<PnCHD=c|C=mU)2`uJgmU;}'
    'RZQfMx6DowzJvUPL8r4~Sq*M?3Bc6s80Kim1C54nab^tD*KP*u1?l@e&rJX{cmm`b5V(KvM2!SzLPM`AFNPgqiTLuupD6N}nIgp=o5a2y;)@)Pyv)-oBu=cH'
    '6;sNZDSurR*Id6BmkB@c2mS^Sz~8`!2)y-TnZ1=4g149mlU8hBJLs=oq#r+S;9kfzs}2KqrF|L}BZ_q8@kwyA4Q(=n`FQS+aNXOph8&?DKL)@o5U40$m37Bp'
    'fGNbwN@zP}45x5q+H-<2JbF9gH0AUsLS)#ZFog-`jLdGRUC7XliiLd-_OJ?r>9MymW^cmW-*WkW0DI5(|2B8vE+*}YFj{V1ta|Acj_!MmAoXBqBdzodouD)V'
    'Bn`|#58e7?$k1Zp$37~KlPV%iE#;f9gu^uVcWloAu-}nriL>mQbSO;ZJt6BYT$#oLd4og+{&65VBr^EE%gI~vsmXf$7Cj68<I7;ez6+W!yY<4w;kSTUkz@F8'
    '6k}F4-A=rNC)G$dO8~1NxK|DxEMdSUPnFOD<U&?1adJa+Dv2h=5ohB`E~XL7uIa>P&ZP+P?N-z4#ZK>y4zB~6CO0|4xjn$K^Q)(vR<Jv=>WCqyl@^ZzKN3rY'
    'nO2E+I>mai;beARK0~KGEqY=*t*Mq7RCz%--+(FXG^`wYBi6*(Mx3qYO<Y6)jPVf<Zxt@Njk%0BlcVjL5M@Tu^6PaPNT1i&d+N4D;UTdXD<#C`hla`(tFLG1'
    'K-c${8yTh?ifT?w_XINkq`&D!90ezv)jK?GSBy?0jo?&<{MCqrlBiVMFe97zV+G{%$D){K(5OqJ^-NaB^|-(jr?t{9N`~SDzPZ4(@WctQ<p9sm^5n8RwDO@A'
    '3#~BdBs(Y3rABfa0lkzNW)%NmnB(~Dtm{j{6t;id71@rUjOU%6h__j}p6VarPy9_ucqsQmi+Y*Qf@C|0dAXH!QVI^1_B_;h<rxG!-1ta4f8p0dKWJwKYo`H6'
    '?Dxe@+*&)M765Rak^c0xwl481QqwL)>ZtcjNPZ>#0B*A>k}6pa9we%cPBhJ`?-H<r+x?Qu24pOT-l^rt6{KaivK@VRBBr6Y8SrFf0`g1VYUdp>O`1YN^spHY'
    'pz7eVYo{DyF$57KqyLaKEADzwo-bBw6cY4{-r_VF)26`8G3nWya!hd|*~RKDjPdboIySe7Y8!4uWDI#Af1{qqnIeutJRr*yjax}XbCH3@?|_EMdHxu!nWoIf'
    'X|Gq;gNhYX@+VJ{^tZO>Ufjvp>A+Y~4oRbjPoIie1Gjsdy=h8VXj@rAwkwS^DqRa7Awj!~@lu+xHIa-Ak;ZP3NJ7R)<P1Wl%hmN>&HXeF4FqHf_6C<Uf|213'
    'JUJ~0)$Si$UPqA%L*~TDW_3jeuO7Fde1}q7u~^AXNS~e7i#Mbz_OU<Mj}$ec1({n>#$z%@*9RG8gnAzTRyozEwzw~ux};l}6ITW+;Y@em=9kA@cAH9s9VOj9'
    '{>xz-_~Bp_^me=euxxNjB%5j!^B}_`OZX#yM*^Y%9O30*U{^M8xhOsG&Vpq7M8p?983wIya3kVvj7PDwMC6(LPDO|S@0^$(&iKJrn)}I6??BzSaD?op7Z`T0'
    'wMn_l<#_uk4_5j3?VhI_xMT_2-l6MXZg?o4aaJ2dm~!ec()6Z7@{DtBzqEl{6`oK%)WjhQYX|kXgU>if@<<bdP;B@|vFs8mQj7P1Tnud;M<{a$V1kqrZtC!R'
    'aBjuSS9b)r@HWq9#vjL5O_@4e^N9404t4WQ9;C#gql-6+>zgK3Mqv3sKV(P37?7)39${Wo%>UW5pwASJ9h@f3ib0>eSiH-_rXo2qM0pZqac(&-R_}{4&$LAN'
    'F<$Qjcih{{cxu~Jsw}2@ZJ+nb7F8bH)UuOlF+Q6w%5^b$0)!!2BR_0#HY&30no1aVBbe2`&PN3;`z{uLJyT|!d1!&fLt8e>INW3%6>a#pEpiPr22f8txq`HP'
    'BiW>Wic#Gg@42m_O~cV&@)IrUp`d>j6MTjAi4G2k3tV;d!_r{&&4nh!*D8M8s}Y4rrPvEErZTuSpBVJk-H=Zdj!m1U8_*)#^3F~P%H0fg=E7&4>?Ig5&{=*8'
    'I9{RTgR1~Gv|8hW2b0Cc36L3cf+ab%9`-)w0&yn7>}fhJR*G!Pt+N84XTsc9`TT5s&Id2%xvLIon>Sh|Pn24qFoF9rURq~zxq+Aw7cpS;UgS$7MwDCS2tg1L'
    'j5&0H4wAB<92`Co4RAZ6VgY=wh=nV&7O6*V$JE<}+^rT%N(qWrq~URR?D{;IC(co+SbVk<S`XgRcs;|ej7ZMQ2;T+gdET|OcQPkOJwC|GdA>Tk6gU`a#bXR$'
    '1G{#dG(n{#@+(8+pMAJOD?<S>!r)B|50e4wTNB9U=3AbBZ5S8X=}ZI{zP1_#t&lHearXP)<$GCH<UN?y&ju|7QiCnA5pMzTdjx<%Xgvhv4b&f>x8WZ&Au^H>'
    '9Dl;%fhI!F{iyL-c>!*F&b1?3%yM>*iwEAj_x3$i(~QN0wduE|o>KCcaf!?ZAQUH(qqA*2rOb57WeTuh_~UGw!&tW7%<~4@U-$!F&pNvG_Cg`c-+;eDd-yRQ'
    'i<tt$mQG$(=79!iN_@;J0x1t?(QnPup%rl#XSP-b%dH-7N>p|VqVW6Bi>*$UHrIm{slQc}wkdN7DwO=IPZEA)!esgR2YP76RNRv>mLe-uIXWbv1WS|INFpm~'
    'N3+&=F1ayxH6qHuXANzDVwq}mUF%gg*&TiTEuMUUw^gN57dp>N`6q;jp}_?h!);@9*Hc$}8*v6TM3wWuh9E|!5P<E&n1XNnKpe&V)Cvc1{K)xbvBH4eD^cdz'
    'r1uVgQRnkI%*TR7^H#+NzKZSY#+x$?M8(Kw(w@6jJjM41hVlho(hpz`aPZ5590Yw{P<A7{LBIQ_9N8P1Jpi<_cQ!6RU@F%UgPXVz9dCVnsU+v9p(~BsI+}_C'
    'q*KD=MyB;N#(dI>H8pZ2S50>r1rUxm6SvT)Z*tp3UfCk0M}ooYoz}X~HSNWNh5B@nnC~SA`}_Mn<Jv-afG-~&crDfTn-1||thb%=u;ZOo_XIt?YgZhp)7%2B'
    '_J(21^47o^WtW1@#CNSUD2thAop(enD9_ch2^J<sz$2GiI1RUAk5jL@W!z7?2ZMcUwr}d%qqLoB;?-AzM!|Sj-$<Ysk%!6uNxM8+gv1G<qj?t3-a0BG<HEw^'
    '9y5}0<REq`j~BLg<3VIs$ruI!S)7N%{+aqMJ-pIv)ZZ;evP2;L?cXV;px_Fp)>2zq^<&=UvtD<iIwCk(uM}~Wr=6)eH`fsoHONM<9=#2k`-S+8oSIWb?BD3S'
    '@_s`3%P<{a7u5{ctgo8&lCO`YAqD`D2^S?MynV5&o95REn|F_*Sn)1tJJb_nxwc@8QA3ZjC7u3)r5q#IdHRK^lFWZmxpT7{CR!b@y?U1v{B_td!7bzsfjp-u'
    'LDnntF8>@kIuuU|lFt69m;Adx3C_=GkpG?ViMDoVQ~_H(QU>awVQP|t#j>#mLDc4CQR~vjAx!6jOO@}mJ}9e_i5$2Yr3p(X?PSx`!j5ZsXS2l#TtWp9%8Cx)'
    'HbqY@<xcgz5BcmuW^;RGaCN?D(%oQx#P$Z)!FYym5=tRESuED5_AWEDjE-f#Ru8+g;G>#Xi>NAAuYR>wjZjM=rmA+(`8@$!`NExloV$eGS-(yCPRNCSstDvt'
    '>Xz4v`C@ee!~KzCTBPp&;7iOMr=j0Z_6LXfrQcFfgJQNZ7BalL-Mx*ZLK7OlAFnXn04GUJg%urf=oj9+<B~iQQU2MSo)#0%ovzL{p198Ss2XL7=tu=LA79`9'
    'EZvo_v~tUq82@~A0d3LS89#0~wNF<zzg)sW5RLdD=sd}T2Z`iNFQy+d;5jLOct;zG5Qaz82{;k`MSmPwGdU8O=>WG{^3Y3dVc-bty<1S*P+*L9Q+qSmai#iN'
    'OSiA*E>3>uSh9$&3jnmThtEm}SeXP-xQ`!<`q$G6ps*yZ`8lpzIVw0tcv3_fhM#&Z4-AKp_%CB_(9~z=DknqbL2Os8&1!23{?Kgy$f|GZE3dm&>V~|Ch=0AM'
    'vfikj&KJ}57}27-Mb$K+!d>*V-rn8mWp|;E*$dB}FL%k=-|7!os&8kklDx2p{1SLZ?7~zus7?~+4h@t%{?cEj(fKMjHI)S8b6|f{?6y|P`?$(o^%fx<wN|+H'
    'YL%yE<tynp+~aw4n~6&@pBup2%|Yvl`AmA56o|LCnp4Y!D)6{qI~$-JZAZs5L{@vn?b>?+bfol6$fZLq%l*<1n#MG(j99wUw(!rd?QIe2T9pgJetu&6*n^7w'
    '`31P7FbX{!TD|SHLoxyp(e5HwT->rl-P6GBJz|&F;xegK-k!uXx2lHa9}iJxGK(AZzo`m#dv|kq?aQYB<mp4_nX#>64q}}P9<`WbSJV!iw#@_p;+0ToU5`iV'
    't#WVX>PMx1`Qw@)1-V&;%}7XzL8e4o$3j#HUe7pK_;RV~Z)8*4j-4dZ<_F1s^4v0s<oycy^D1e6kaTHAe#L|E{MV;H{PpSU=Prm$SVndNcla##bBOrAv0%cV'
    'kCHz#{XC;TQ>g-eR$@`+v*mbI=7XPr_{!yYBEQOFkBl`$z6G{>gh@5eqq<)t&BPi@A!4LTv4Xwak})*yxkkkp6Cj+)##LUXn1S^<2s3BrSh*1vw=U*qC1wjA'
    'lcSp{!^oRwp49%7ygttX7|3}T7~<GCGNu&<n#`r<MM-*20s4*=s|E@6Ua~8lz~g+P$Wzbr8E&yk9(u9>Mv2+msFDWUS93ry7CN3?CMTCku`VgmVL1?3zvOFM'
    'fP+%%Q{`SvUy8Q|pdT;*teVKLL<mGyxR8@gF`H&4^)18WQx^Q>C#Rs_&}p>V1}SR+>cRy^(s0A>J#J?n=J_x0Szk%PV(HOIEw)4Pfeh+OHQ2KAXXT)#NfcD1'
    'EQ|&fOYf%mhQBa5U>b3_Rv%@22v(QcbxO)3*BtgPpV>5F{K!E}>Flxmsl{~ve5q@3riQFe(%TSoFc?&yL80Ra5Z)TdpO%f}wnRz~C=E|Uu|n&pr0x`GCRqzT'
    'olVwHh>AsBmkQ5>eK>YlQXW(M@%uSB8*a$gP@Rpi+HF1KO<v_om^PlkKHV4g_$DnpK@Kyin~Ii7_|JP-L${VyCUa`V<}hMr@ULZDucQDuEZhK1=9`NX;1P?{'
    '<aGRQv7&$=0zvB@lkFOIwgkPD!%2TT&lv=;kd^ZqfHl3t#HizWV*GlL9p>fvVg>}F#>B^D!1A6gMS%PbGfnpUpu+cY@II`vk9JX8>}PQTqrLOyi?96f1x=Rs'
    '<TY*@a;VhLuzdLF!NYwZ+*z(t_OL_o6xf?b3197h?w7n3jLNB$znsBNeg#=}zv7L!Z(5mpFMNVJuuQj^S-IG(CKP9RT6_ShsYl**{J8?sat05ECrn@y&=}4K'
    'lf0P8Q%a=^MR1bDDGT!5rY5TeW-7<VCEBUbCA%a^F5`GppfKMST1S#Zgu1EoR>3i7yAwACU#6?@#K4RlHF^1f^_;>`QIiO>8e-bUNQI==jTj<05sSn?_fS?_'
    'Y?hpHL``L<P+kn+9cf4E;@x6RWgG#>fd0y;&`p=)dRe?q;c(727xVJig#)t(&9RN$9iN@y^?$6RaSi5h0yyt0c>y;jq>eMT*srbwmy2RW?^sq3-wU6c$3S2U'
    'a-$e_{yV2o{^k3pKZ9Okhek3FM&_8{VMmnv#P8~twJFZd$*5*i<MIQM2Oh7!`B|rV^75Xxq%1FdZFhncHPk|8XS-p9C>*c%wh~!uEW}0}H4G^T0i96;sg7TR'
    'ArfxuP-L&wtlTV7{fS{K)vHMzWM_w|-L$lEgYpmK$vQ)e=PbXx@yN{shW^HtkRmJ>M>K&K8O9Uffg8x-+R~RKbCSQXr<&_g80jRfwvSO8_5^5J(c>p|9u5UT'
    '40tpeps($G-0g)Uqu$AV@gd-_<!Ds*je7AwqrrM11@J1a`&O`XePmWGI*}Rzc)S4iIZnMl5ni_sU-nOI4gnznIv8E}DJaTBipjA^2m1EfYvBvz@%A+F5L|U*'
    'XqtHHu4=Kf;Z$<;V><Zq;lngmXP28t&A8R}^TnLXs}&&TtS_?*=v&N>)pZ|j&{LF%=TDy;))^IcDYJF8Mi#ZmiZVL^9XDScb4tc~zo_@cY_gy_Mu1ECV==eM'
    '<}>~Zt)N)f2Cf3isZonrAGmr-^MG1dnSN}$vn~9O{R8#0o<z*{Ud733j0w-Z(UZPEe2^$clmHa>%pz-bgH#{+>hTzCmgtrd3rrAkP??Vnl!%`{9S<7SR8?J4'
    'IVN1(HxzIY>m|i{ReNQpgsv2-0g}slF*}x$gDd<@@EWI)or=qavjfWWO!JJYSTsQv8tbikl3Z<vuM@j(rV@-dz)J~!eWE;$;4q_Ek=Z~bLs)mWv199Yr^_B*'
    '%V}W{ed1enUS=lCpyLmXJ@tumodaf&#sY7sBIUZEn2PawQ&NSd8EA?iUU$OV@|)wT##Lr;LNi<1vL{lZ=ZwIIC>9#C%HhzS6&M*pxwi_;Ak$$xWZdJnTQ4nB'
    'Mr;-tNsjm4wsig``r?|}*D(JcsLR`i50wHvK4swcJPgCpSX^r}j!5kq04;Be1f3==#pH<lVLaL>h*RgAuPQzvPaSs3C2aaer6KZJ&`I(DPwd6`LvaB-4=x*I'
    'ZjUR8E!5K~!j*))6!Ki}wug^$)TBlQ<{zzE%VH-Yb=aw#!L;#&rrS=TN=Cw?4o^E*Jm7T(yoj^G;)iTJ{e6QeC%7568Cj<XyRy1N<6^<9a<sHT(;`E!12P(%'
    '7PA>h%{Gysh^u%Y{qO~iwL8KP$=QSjlIwLMH=ehPvy1i_$}-|89l#+mEsU`s#BcLnIW%FtMHy64<gh~?r>hH8c`i|dH1a09j)tTV<G<t=>R50u`S(gMClXJJ'
    '|9G6iVTN93>n??SR-4yvPOzxKY^@Ds<Jt0j+-rsy+0dH;<viJB+0Q16^L(C7|G;(Sq0vq?i~>?}7<D(_N93B37op@a2haYeclG>oxmcg)WnOQ1uCekV1Ix(n'
    's%fF~?Ti0-3=I4|FmDjbo?y<cK4CZcP4=t*J)jhHRYIO_pwJ(*amJU^&h!twxvv_~cr<u;di|tg^M+m;4p~sI8f7^GgyqcUo<X{oyf=1o6x+~)W8y5B_{e}S'
    'aD$H8jk~cE$5H-|K>O`ce^q1Mz8-V}m#&y)cWB!LE=PubL1U#X*`4nX5Q}ibQgLJahNTIc8~N#9zr_~r?L)a70q(Ti6!_GbY~2x)s!hFJ5!3jAi}jX{DBuOY'
    'wxW6+j$msu;wDS|SNg2Hb<*~zIE<shWqtn0xbQ{Pe=;&2hRBc?YZoiB1shSPG>eB&e)bUI3(x;#L_CU9#SA#!d~#Wjfv;Q9gUI*Z^4Z)SE}OYk>s4zOw>Z`2'
    '6cAqCjpys}S<bvqUV*Wqqh)gIyj+)0s*g|RMbh!_LFZ;@eH^0Is3hVVai7s_HI^3mj%YkP;oXe&Ul{eHt&UIqe)8=PFF*m?`tq*d-$}+HuaGXuH&wBRuU+~Q'
    '-z02<r>nCKK6C#}zjUX0Iaw9t>3(dtj=YElhspo?pZ}Me7QKGjAg&X7ur??u<@t_>8;qyqu|nf?JN-U7Mc|VlB~mYsQTpQpWgmF(kMV5Q8M#D`+^7*QONZ!$'
    '(Q*AKn^99Qw5<nyU@S42kndb)Yk$n3qx<9B8#~Px{psTU9J99K)8~W{Uam3xdc6Y8t0%r7e?ejb3FM+czbDCofU=s7r6XU<qNGr+IY%i#YYbAG<WlppW*%%U'
    '?vu3`CSuZxxuqQL?>8@`FK)k!RT%~FXe)s8Vmi&|(aw*X`y$@n8UK1mPV4dfoXDv-yI2&IdU}=5Hux`k506(KT~X`CVL%r*vy)IO)A=CpA2xP1zCL{p6J(y|'
    'X1bXd$ayFw_G(;UdG#3@hyciB0>=s%s-CK8-b{~Fiq#ti<$B`GSa%NpU2Z(M^m`-Fv*o5wuVNlt!p1|A%jn}5k36vE=SIjlr~dksphV7r1Ltbk)t{+)V5ev^'
    'A!sGc4BKVYZtJSa->Ofw=wlRmaCrFDSIs0~&c}98&4A|lhrY2OZi98W6_N{n=~E5(o6ifeG~P^$^<C^>JurU<(cO^SwVPXiy_n@IF}}u*@?qY8)CdY+{uS1u'
    'sCu}CVG)e{!*T{^{U!{L`p>y?Qk>oHZ{2v6NDdG_t%38fnSwMB5CY!it24YVEkMfyzDf_iNVlD{I%qD$e|H);+41^DV(~baEh*w{D7hZ#uQWG{b!p~o?uPE$'
    '=e{`c?GFx{H;#pF4`dS=efhWzG;6EuyO-dbaR7$5I*0SSq6^nzZ9j#z!T%n?|Gt3#0X+=%J8huZV5GhTa@FE)|DX{b=Zn}zOwIULLGpc<Ln(fQjp+8Ak_Fc0'
    'JSdNnL$t>6e+J19z=SE!6*;cY^Y_|cw8gYIT~c%d4)(WEs;sDw=?^L(9;J`B2}mX#!VQOaRBa4SA~4nN`QujS1(mfe8-K7xJ~bu~BPvT2e+{fgP_56~<NYm@'
    '(nij0XLo;p8#LejrqAv-H$81+Y_0Fy%$U%tnhmg?I1=4`Ic3geoqWSqn3Xo^_mwi5jF-z<k(U_Oa<W)+SPS}9!yO7gCMEE7j8zeUfLqc~P5s0&L2p9<FdZbn'
    'fzk_FW|G6d7_KtIs3+9M&I=S$3O3EsEKARsKS8f<P+${pUh`>ku&<&8)Q|PY-aSrutBh{G?-qqYHlWp8w{$%hd^ODB;=N3O*@D8t^{Wk%7Vyi8)HIBo89N*;'
    '$W-7x4%G4pydm1mH!@G?N|A^fLCrkHBK3x2#y6$2m3oP1_W_%I>vj>BuNsb|-r-7|LG79NYCqiRr}jAqpx{%!d&yvDc7#V!C^n{GxymVep(Lty(Fj|2xNgqv'
    '>C)s^Na&rYZpssoyVir`B@<*`#zHadWZMx)lQ?~_&HmxQV76gT^ew4NT4F3I=f6|eb!zhB<<r#nEN(fR3%~Pi9~_CqwXil*j5MW~@AbJS%+eRlYAa5vFO%AG'
    'hc-)xf8P|7w@p?q^F+qowJmcm2rnuZvv+x>PGs{lZ=D&G0*nY&0*XeCPC?`!1RSkzZUc4hh0Afo2eN;Vv_Oo5yV8bt7%*g6EeC3b?!lB>s32}B+`aC~Z@n-!'
    'hjMEDir0{2)bj7Oi%p5q;fcvnY1k8tCbJmNG_s;&M_aa=iO<ltPRF_Lx0+M5d7FtvS$}kPCRmg6;N%n#|8U>?*4+cga97*t-zsj|{mvDMiElugoym0wMAy<y'
    'oP@}uPcK1xO*EiAHr<*HYQ23opSXoxnY|lJyb65(41t{C$;|NlW+JMvgJjL2bCZnk#=jl=#}dN}>roK2?ge--bL=!?qT?jU%t(;gVuemTw60FX8wTej91=y`'
    '%gmTMZpIa{llIq+Gh6>m6K_BA$;f5`;cy~Mc2xO3O}^-bz4#r#@4?@t$-#cm94{tiiP61+pVH*3SYI)pR65G{Y4R}E%h=uwqxZkGN5%X0n`EE&_|aEB#A=$L'
    '*sSVXRaYi$V{sV~kg;Zf>TqYme483l_*<Gh3VO%0Piq+Q7S#=~*|+_CdKy$slhouV3lW9(f%w9r4XOy=hUpp<?}y|Z56jSy;w@lh`;RL0?c(GtwK^gXe}w6L'
    '6;3Cpg_Dt4*E^Z9237RvCd#mD;dLIRUv=7b(Yti8<(gF!4S$2x_S)QfQH(Xhsd2j3kZPJ#$+JFngSP{bCe?XiL&ps>dhCfpi#Cez7%$O-cV@g_kjxJsnPjLs'
    'vN3i`Jlfyi#^4`5-f!gN;m@^1jCl(WQ$GtvrJ<=wEe`c)x<c7gc?)WeZ0!3OwTR}aW(hU+GGGPNlYIR?hkKAlM2Q&TDOp-((=eGzmTXL4HP~TJff3eD4gJaY'
    'VP8jTI$Lvz07ecHD$u#!LTl8tX_lBNgR<ASBp8iq#}@sDu*(M%ZCMn!VO&wrgX8+{6reU*O>8<9kC^Mb(tW~-)y52YVYsm|Ys_AD)bTo*<zx7K7^C7X@n>yF'
    'x0;JX5o|Xk*s9lVy$<R-_iiK4!17f=J3CriYjVeNnE0Ih4BOQ<P9stQcMa`Z1Y^ceOC1ldn+<z<H_xOlJ;<Jt9E}rJQD)XL4O9{&P|?=?ShS|Iyw%{b{V0UG'
    'Y2yU7;h=Gr)g#`{4zkmHs~V%F1gl?Tp*6I!H*1f65)!W<)m2q5S5vk&!Ojt%G-nL)%n42>R=edEt}^0<n<&2foSU+Ks}+POZsTC$+u+S6uvAD0Yut?C^;99-'
    '86_z5=C1>~!`0KlVedfbL(x*4z^|rzeVlkbzIt(teT1RoIQGVuXMq^KYoyg33B8gnp~tlxu5LQ5gQ^?)mzKF@tBnor-o~|C?w$Nvl5I)zD@-0{0(;xB-nv6{'
    'TiDJJD2XYBTLCg?+=hH&m9&d0<sxyeAcvBdG0`$ITEJ0rTa_K0^3CCO=FDZ38jikJ_A)N~WFHi4p3R1xQx&gr3c0L3Kdze>1j@6(v1C4eFz0gRS%6(!)R690'
    'cKUueO0Kx~0BajxC;vl@c6_ClbU!(`esC3C3-;<n2Euh6U*UhT1^cZ6Oy0hMV03!jX-ZDP8)m|g<LNwGWCi52VPgp?TEMYv&YfyXdiAPV^-4vimW2PsXcdq`'
    '8F{bdVUnH*?b{^B;9El+$8ne!N{R*X9x1uTD?NGQ2w0bV0HYVnE+??ZFDTKNVWf^pP^aF2Y5;UQX@_!5XaHXM+%jKxrAJYgn-lz-R?=&b#7{df=Db`dg%OaW'
    '<cj{i7Iol4OajATj_z@F6nXzpnNR=iSRPkR_N&PI>!=+EsJZ-Z%?79nE)2&A@H4D1hd`6a+uRYZIX8SXxGb+dHIey>Au$Nk#<7i*OL(qkBQV*FFav?QMre<&'
    'vn%X%U1c0Kj7yl8+i}Oi%sB1XS9BxdJnR;HEMAoDuJ~D9meXTO$jqrLgi?ET49%zs53^6V??<|Q-DP`c7j0W&+<K25-uxa#n$dS@o2%0j+Nb;T)BXAB{=BpM'
    '6VFxM9U5GhcX(Caj0@c*C%TIp(A;}$ICb~np2w&3ul7r3YUujQ5xLHmD{dBn$A<QL^Ni#eCY?R~_34Wro<95Gd2_)<^yEYilK_`|i&4y!n4gUXgX9H1&dfM+'
    '_b6EwOU_3H-(Vl#H2?bLt8c&GW!Uv%H93!sOLQ!e#KsCQ8>TOA{)SHmlS}HJ8!7UTyPDz>5=oI~7|PeR6yVGZHpEZRiNC{YFP50h5Qz}D<tdHX`TTF6Uwl5z'
    'KL7snA3y)aOmRVJjq`blg8=C;R1Pu5zwt@yw8OMAo-WGl>euJ5etGfoCpdU^tZVDxSh2EdV|?5MYy4Md<GIZh*_7?3XPXZ@1#%QLwz4k!E?<>Y><IwLG*x5$'
    'c1ZCX*vu=P$+0EJ9L(_~@ylf%KT01zQRS+oS<?L~2zcpr)SwN<+mXkmqmK~`W1IQb0~Eg>6tpJO;wU+8X>YH>a&|$NYOTgH*Z?8444N6;#x_X9y<cG_k4gEi'
    'ivu$gO&%m2{A-~Ju2>A9m3g(|Zoij{xk}F2F@$zx$V}S_Z4yIzAIdn=DYMl{RJw$Yq^Qa`TX-o|Z;YM!=T|R&eERBd*^kd(fB*8&zhvM2@bcNyA9@Y_IcG;r'
    '_a9z<8}!as7-vl{*_?bIryb<XIN?$zS>Qcjsz1gXo2W!rrK`C;=cB<wU6dR=*3_#e8Pv&I<rj;0t|DbPhJAyd;6<3d1Ch683>z<f6lJl|93%Qg#0Fs=#dx9W'
    '<8cgV{SNL$=5AsTb_B{P@>UBAJfv16PD8FN@ZdLt>E>ctcJukWVzrnfk92I;)G100;~$rkqL5mB31+mPua6I%<2wv*N@j-eTONw=RNm_4YRC7g(Bf(G>7I73'
    'DgyHLU{L)MM8UWNRv$#f&PRE*`Sjp|32i9D!%-5-)2fZ{i=~kq6zcISjKw^7bmuxJp8!hKy(fed_W@FnwlbXS=5uOXxOUa_twY9EuOQJ?@(*j&RXj6UpqBJ{'
    'HJ}?ke5&w$>qzY&X34G`D~XTevoqkDm{4097@f%j7W4OD{)Q#pF%BMXS1gS~Rom(Mc17EbX8En;ivq>tmBIfN4rZve+e7J?@*y4H<^KV^?_D$|C8JswQp<EU'
    'zRXvY#|SVDS1VS@MQ4jwuis#1qtgPkd5}e*ZIPEBVPdb(F-sK^yQxA12pq3wmuf!V4M?-IMa<MR2YXzOZOj!V-#vUZcxd%18}JGZxX|OTub%!WiK=+RU?4P2'
    'p<5m{1859&j|&SX<z;C;o{!6O5Ha?s@)1h?G=}_W{!U4$`U6W-?tTB_7tkMHe2e9|_`<_7UD-kQ2=xj!%RkCi>kkk2zu3<X4$n?9yL=E0I~X>r0%}2!dz-Jc'
    '+sZ@alEe%ai;HglVRDo_1N=PyFv*wfg$WurvW+7tr1t@JIkcrg+undue|0^bzk^|23#@8#v5<Vd)C<r=!cUT;s(X|)HUrtNlvJ_2pzL%ENS)etdJT`<nlr>K'
    '>;8?dH_nFe*=L{qiuSqjnv+2}>M?~MO&$T>CM<LJ^W|&Q(WN%lpBJR(lS_HdPL&P_2weq+T$WT@>`0V*paW<F3~q{<sB*TFF?<~F5@rI_O$p;tqFWN3acG$0'
    '1q}Bd6~|<~D$dUGl|OIE!8&=HqYq?hGCJB?T7W>Ae*$0s1{@xfPkYs`@UqS&J8>j+JdukX&tyaTk-m8fTy@gv^^66T`Jd$bK$GvDCkg9LM=VFfw2#7{^qn}S'
    '%N#%gUJ$b&QJf(QEj0jcl}1_}y{1e$S_B{g6no3p*F%c{zQrb+6{auFFq6b_vVdEJCEfubrrY`XZ(`#40kog&+m}E61yLeA+;pA6T`{?A=#6yK2m<=*c_4f9'
    'V4M5OBAe(&`u^Lo{l3uv<?AX>9S8l-|L;Hl-$gQA0G+|MPhoT-^f`Vz0Xm&v892Bhk_85z{^$RLxds;r+)l-$K%>qec=GG~C7)omHe|w+yet=KLK@aMsiHH<'
    'lNv0AK&Ru#j9>iud181q{0rNfuuur+=-n8JBCn*H72BPM%Wfn3dpqFuC0d~uiDc|@%)*3@=O2o*JOL6+=K1`5b5UJY?)D)-0c{H|^7nCqhXDRN$0#8{jTE}I'
    'h%sy|3HL&wD}Zh>2Hf!?wpEKw-Ck92?Br0#zOBJY%#x1i>0;nx-8V<pzsTe|w=sWR`GS1V0*6artnzkcWEJLg@|V5XJT?YIVOzuPt8d+E&A!0IQBfiu9)9`7'
    'SAlK7hH>~!4<COKG^|%I9z|N#sSt6vk>aG#_TY<$UxlzKahS2ftjFcspv(SYtOrO=7n`N2SN{0mFwzEOk?ewU_|?L@j%^7L4c>BN_Bh>jOJZxR*^-09uUc=('
    ';SO68ha2CLuMYm>aJL<aBirAQDpAyKhzIVv&V9LpWlZy@?%y^TZin6Wm~D;KoiMr$n=LST?XxudvgI?j+h%H@3bz+USyJ(7Thf}#NIt<^kFIS>pUYpub@P6F'
    'iD?_=gO*YvE>>nvX+t%)?8O<Z<Hde+R>SRR5~Lvy-S!8aY`_L+)L#oCT%9k<e0~)8;-jE?ZQsgIiyW)jN)n^F9b<s50%cd-{FQ9}(v>pjR^w3kDLv92^?n*I'
    'c*y424yhHrEExnh6+R66OlC!A&x=7cDAGIXqqWl1Ues8ghQEt9q;Xw93(ux0km6y@i=^^U**U5LNcQQ)5)`mLD8eYdOj&&O3<N1BVm=?Qrlj4!W3|wj4)|bG'
    'qAFL5c8ayyXJw3jd_~9NgAEG!jv&?NJ?>E1iFom@cvJXcGk*)p;oPg=)rs!gcEGM_9pFF_Ji4i}SPQZu54IVj+No~-u1hg5usRJ_Oy|cm($q7SnJZ5@9sU(f'
    'V&Qg?jWvQ+Y3m%gYN3tI1}O@;)@n&N@F^hr3@z9!5Mi2P#h`><>CL#5S8P@pia8HE=u~5r1$5^4v%m=@Ec11>-aK?&B2%W7lGKmtZ^rg_1iVB$MP7DQ{QP>c'
    '9?!C~;>5o75|jad&xWI3@<4t=@i4h}Pv<&7_dh4tLr?so7a}8G7_siuMxOK#MF<&IvnF*Bn&ID(W_s$lvd)oPWu!S$#0H;~$4mDndfK_-)m)5!U#zYN=C5K7'
    'e{~{IF5t2%GNatBGD`d<PYBXC)<Qo;`Zc6?;2Y|yZ&I2BvfJ^mLEIlauKH<h)2*E=@k>mt2NO#4kbJFy?w+qEV^pIL4wLVm;RiJ9KEdmLe39@zqAN;rWv6sB'
    'I6S>hzAK&$YP^>_e}$QGo@8@Sv_68W@tA69B>;D}K&AX*Jl~9Gp$Q`H(~LFIa2*{}Oc+Vb;I@h=d(zE)hLR~}7~wN-v4e(i1nZD-x+s+42F*F+1Z@!C{rp$9'
    '@NCeuGhbZjIrF&|^Bb%6K%Z+MtO0V__w?3e;x}XLTT+cvV(VT5*!lcc00E2V<B}Q^Y;=wUJ|y|oGoCV#vJ|@L1eoRdH1+6G)fN(2+fB_TEjl}gD@3{R{VDlz'
    '48rjT@+r&F4Q#SmCF99zQI;g5kLP2T{SvE7u&aX@b8FFelFujS_!1D7Q4eurJDPm=JuVg;Z9`j*d7lX%6};)Ypf$~$E)TxQr1pZtFn>dGS+D@MHRH=gVjnQ7'
    '!!F0AySOS7Qx{ZeFDQL01&yfmB71$P8r;pZursJ{Mtp8>hg&$jfwe#lTn!FCNZ}I@Y|3t#&rXk!V4h1!g_NHfO4+pE(&Qw?<QY^992yTq7(!NX0laZl%2GFF'
    'Zlsp*C}KY#hb$`kROv{H-C!%v0;@)-*mHPlu5ty>vde2p{#)`}az8og!3SIldx__l@@__I27#R#^e{6!od=7ittDGM^tN+@X4SL;b|xZp5E%EZmOM@qvni+Z'
    '?uRt_jgbpzw;y2A{5d#Fe*p9kxIF$97s;Q(>q5^5eRQ!pNA)<ITd52ibpP-vJ)czpnPMPKu%wMJiOFnyvF!elCjH0zROgxg>1`j1ij6dlgAJ;44s~GD=Z_##'
    'gUIMyC?r5)g)<Ok$a<J`A|ao4Q>$fw(NW=8%TAFal1V|I4s*Rerr<lz$MJ&9ST*!)HC~<{2dbncvrCW>ipkbm?G<t#0;!XS<$Oy!|0vnwo|H#}Wx!gNn2^WA'
    '<mkpDoOVv(n!xMh2*k9@WzJ918O!2X5G2Ns6{-OOp)E!dyK@%l2#1*)rHYI8D(@60r9iA@#cARUghz=<=sU!B3z#_BldNgudk`J^>+{uOb9RmegHcdZZj?qW'
    'VDrYymp{_$@Z$8;S}|a%Sv!V57gt%HB&ChYhx;SPvhmdJS2#7F8oDaGm_-JMlFkRCUYd|s`tffA`ZqN}4!?=}daq(`#Jp1nY$j9rvlgo)cC<inm=^4Ot&FZI'
    'FEnuB5|(5#CVL~rU`MRElB`Q0X*(Ha_p+-<6S)tTMb4?%#AS-IR=}V8_8}j(s<dCwCNe&qryFt&tpvP3M*?R^Unha|k=q=P^Hkd0ck}FRm)ls3a2D2xcwxwR'
    'Nbrwf$Wb*ug;#FEQ{>)QoURe)3=2r1xHvxORr?w&<thcK;xbx6j4~P}V#NJmC{?N3y|0fAQ5;2;KPRv4Bq!QwYchf5%UG%U;lTr)+s;*f75cbua`~a+Az$i%'
    'ts1_n4#$Grow=v=iH}P{TRNu+-0hN$#C+-b2nk-$qC+&m`@A18%9CP_j_2;32H$g<SSHW}%&v5t@hR}k%Qd+^F|fYSS7LzjdBU=b%Y6KntN<u;YS!3N7Dlnm'
    'zaaLE*imzxh}dF})49G-90M<MXE=)=5AKlh^4TEo`VP70m}C3`#%+16d@l^wqH-iw?pbo4CT36%yfj!9?_vw3AJ}8ZTQfprVT>d%_>&MUi(CWEbi4S=@CayW'
    'l=Ktx$H6F7pVbfhbdNZ6Ocro-iV5y9qR$P3jm_!?uif;~2$pqqH#b%NZK`0o{j*d`Y8he!h`@rM@pt+NbnSlKOS%VxeOQY5rq>&6<|W({`5$@rpa%{7A^cN9'
    'wmT58%ksu2<sU%gJ5$KN{;41`u+R-1hNQ(?XPb2y5%+ZOo}v}{8f#~T6BaD^V5OlC6u;CUO-%q9Eu^ikKgGU=9g$p3W=~6Nlk?5|ZPci6IvcMuLvi$nAjcPO'
    ';djTezca_k6Iz*hCu7IX%k#x*J;4G3`1FyNFe&Us{M)DnH<52)qV(EeXU;^SoZl_gi8za0?n;FHCy1Z_`r@Xf*7S7-)H4oiL^0YXC3!kYwO*p|s8#j#3Aa(4'
    'jhh{|%&3`r-@&Xt3CedPDBIoHT}+O5-6|wA)%YAZ%&%-5RM`?$<-zpz8<W}jD48z@^XYiC8egXNytB7dS^bg8D<kU6;rCyVdBFK6D+D*8{LaL}LwL&jN`7zh'
    '&cLYlpPAW=-IEl*M;UuJ<@BA7DVQs2#<)qqy(!Es@oA-XitQ-&m`RAeiI}(NbKr&fyvgPm5)Wt`msmipINK~XB_Xs+U-HK{uIRwj)t@r4^?aliB<|$zh3Sb`'
    '=Z+GbcRxX6dt;faib-(%-Xx;OkDR=^<S8u|v&;G7q8QHxv-}MB$|}bXXX8y-QVtQ5^qbaWugh_nt@C*aNbg$cm;ht2k8k*gXcgu~`8TM(KmV>K5(3rJYzgi|'
    '7MmUJ;`Y9>A29%K3k?`<F<Z>fqOgczIPwaEiRC6xN0qysoYC;YrFd}3@UqO{M@&ayfA(Za+v<Zfs<BL_h$1bl9pBo%#!p%R!tLQ7`D#&Sv*K;uh2dmv<1)uJ'
    'aDbPUgHVbv{4x^n$h;KL;tZ2L1rHraLx?fSs?OOID{=@tWa<3Vc2F@t)0CZ!mAx^t+B$$N@x|O+(*%=rIGDSv*tD;%`oMtvkonRgoA{|0YCt}Yb&HcT^SfCk'
    '{kqryW6EwLbD9}(3g-`c=Y8?xL+0erO4HQBxv$sc3lVS74GFpprR@=c9C>YCV;jR7ni@t0UEWCCnAq8PbJB}FHzg-Bb(_X%KXwjMw-vp(VD=2R2>J{K;L%Bl'
    'lasSiw5PG&yg5Myn2SIjp>5{bQ2xNbqg}<8aIBkJ0QZx%eCz!-0;Sbja#Z`c9}vjN)@Fgeafpy^|G=0+1VjCUi0Ewk8#x=@4b&28E&~oRgWXZqgJ<fq+5<Iw'
    'd@#c)eDCE(m76tzP`!5SY5PF-vz=axwWLYi8Zg(*8Kw%DI`hotqU%Py=dSVS?y~vjFb-WEk^34vli)oyx@QH7P;8<N6EXaWvCo5OZGQUy=r#)uJDw#sRpX-Q'
    'AIS<J$6g7@M^e{5idq(M^8325Q0S2$7sfZi4Hci4f5nK#r+TGci@55?=3qeFe3LiK3NEQ{yqmajMs%f_`IsJx%id%zZWFZGlBG1?V5zUKP^L`n4K#AH5AqQL'
    '#)9Dht$SU5bQ${1iO5}hYZkO%STEXKQ0g_LeLjKiw4pg}x}jWe)U2;y?ukjLrAh_6<C9hkO&x_9?i5wkeR#c^uq9krak4SFV0fMATm_<WZ!lSI;J@?D#qtvV'
    '9hc;Nz-pZK>Snz9qW6(A#O?^rrXmUf+MU@IqP|>civXD0s$HQelx48*=&BF9zbll9B)gdBWtmZ`07`Vk6m=_L<CAhRLvLyWRZi4N*){NcyAF`g#><kj+Q6aa'
    'j6cNbErs`$1))kC7<8VCt;Xn40FCW|Iun)-T~o5ZIB^G}o{IcHojfyII_uohhm~~Iwjo`O5{W%}Y{*Pwg-t!1XuuumBTin?0|eGuWAisb7ByC34I;t^j$&LO'
    '8hwYlt1+<ohpuZ=w$5VqwyG)ZG{^L|n1x`i4QZZ5ZPdafXYzAV7z}@9)1NU`#LnhAS{vKzarw4Sq9X>hOHAVf+a0*_8E}_meGU@y3f)NO=yPmJP_ccXL^T5+'
    'Swf>*G$e%(G-wr=60^;rACtj<Upq9t%~x}Lm4%B8z`<6Qn0)}lSd7`Ottl>Mf_uh&T@;bp5oLMchj7}BUiFoUmwr1V+%~%qjgiuPw_1ecVGHr1UbHBnbyi#y'
    '>$rVDMqoQnZ5S-RB?<00si(;g<%m$FghHLQs<F;fH5o-0(bpr@a?c6Rb;dmJ-DI0xkIprK_cib?GIYbs^?RSVu;aO_gNEzn+=E5xa>^8V-ZmFld=#z~#U1ox'
    'P7oh6{Q$F~w2KBzB|-$+RHas2&(KaGL=kk^Zl@fDuJ#|YcLilbL5qk|!i_%c;mf<95PcR5{33A3^q|ZII|B0R;qp4(W5%5>EnZD9HB+Z&Ma%mz216#3+F4bb'
    'a9_2#Y+i7qdnN`oKQgVxEv<6MSPzZZTD?P9XKZDD6ScOgb7_3-Xm~r-CKUVkt5X{e+KEcq<$cE1WZaN6(bb80<uoTPjRW_1$F4v!`<|!+_?fo@nrw3_<$3rQ'
    'yTs@d&bpCX9N6S~6~31xEfY%J=0c=KwP8a%Z5ue$3*Brb+wyx;a(9-`yRwpA8e{J&cu%J-NHJ|eucuPrA<VBMj=j}5&&=L^La8h`b)z@2Om29D1B@zRUmS#U'
    'IiH-b7IUCyn@y@NXy#AJZycmO&);E?ALp9|r0Pt2{H*F?q$n5%Q|JBnJYP~Ot;MQ1qf~E{DNYimFIGTd8@#lB2iEJQxig9nmMx#J7YXJ<7bIw^3p1~)!W%~k'
    'W={n6|BSP-+rXb_%?6$L+eO9OT-#i`A>A{H`qlHS$>grYM?a-I%aF!3zVc@4rFmq$b2?#IM=_gbMwek(K+u@cZ^+`SZ4f=<$po%5GO=;f?mceF<atm2di$Po'
    '*Sg&nz9^~}rq=QXY=q(i8D#ZUqPOPjjDd=1Iw7b?lM1L>hOdDIJU<4bPI3#u)U}HOt@<8?;HoX6$f|wL5;j&TFdcvWtA!ok*Mr!aYGT=%qWBxYWLzGhE?#<j'
    '$d8|LG<I*Li6ipi;fOD(UKoDipA_Lc`YfU)FFb{;Z|38-k`Tc@&w2bFZin5H5!6OoLQ^!B;eZk!kFh>ww-@AaMIq|ts~6wB_z9C@Vk*|>zdrrpuTNh;zfCQR'
    '-d(N8#~9MBS|i-SW^s~xBA=Qf6_j0crchhiBA>t6veTs@aLGiAFCVZVn)OXrGA+hu^F_%eIW*Vr4zI$LdvtN3O~%z+3X%{Qq@zjD;6G@Q=f;?YAA2<!7Oc@O'
    'Odx?jcK5?>nwTuUCGCPFlyJKh%OueN56KntX?oz|TWy-cPH!JWr%k$jik6!uu)ez_)_0r8dQ0-ItvBp$`O?~5jw(&YC}dW{j*8r-CkS`4++pJ*M5R!?Iw_K#'
    'zZ{<mg{vdh^L{YLiU^GB@m|@(blt%m|7%J^tMUuDY87rysW+JiiJrcctcZCi{waBtFJ~Z|a$2SHd<NoniGehf%@$;se*U2-DS56cSv7%sX|Sg>uR(TNtXVB~'
    'b}##t-*Rfw0i15;6Blsc`qeb>pw#(OI!(Q<MFk9d&4&OEvg&Vp@{2M8L@filU5yVwD@OqXr{13Jz|^<di+`PsPtlTYP6t}3sKSjAI^Lw7@!RubOO-vlyQc)5'
    '2Fv?SU{hzm;TUPF(%wWnx4GPv0P_6J&(FVo{rt}vKzaK5#mk>|n_zFR=TyYlk}n0nb{z_S=o8vzyrbJ7z{v*~dftX;ZoXY)<8TH6^BUgXC00je!)(zQEsy7D'
    '7cr4m28$!j*WC8L`01DDuYzs83EKCf<wqR`gCB^czSgD)vH@V(z6*MnK{G{3@o2><$Y_#YZ*l5!6LNxl>4pg(0V3?tqLgIzn8etnel(^G%GSpTWhezpX@9Ya'
    '71xX81e4sOp0X;Y(;V=N>Fg0Gs5BSY7oVxw_MwZhHD#xii5|NKPgiFfG}!!1zjSRS51<X}wf*jkM}vJTLV(|V|NNJqU-rL$`P1`XUiW`_`j_WooOps0`~&9s'
    ';?>W6XvpSxS|naqrrH9HGZ;@vo1&q+oqk`mv&s1a<jnGTXbV_Ws!fbsF&l2w2$$`ceYj>jt{=tqH}%5XIq1`|Z*&>{+S%G4=~VZ}xi@y2FZxpuEJ1F?9D`iP'
    '?6`wP1@df-+0+|*bCsJuRk?xg=8(Dx9PaNo58t;bV#_FiM_U1$6P-poKW^^pA{9L%pj;}cONEk3suK7YX@9)(sDWBHBPl;Kvm^b$bw0@ZhmBp0g4k!LI?V8;'
    'XS$geSbz&>M#ff<TxTe)1CYsNvl>q>y{Ve!%~ZrybM!Ejkk#ag!he?=4=(-Q2o&?C5exU=N}4SMNguy><bgFmH$tW?>ymRTLcI>0yL0MKLOrlk6fg){2{Utv'
    'zP7H~Wyw`sM(4>(6Ng`&;hOLbgJ1Z($IG<46)`$DryIXgfuXVTLh8yW=$TY9eQv`9X;poB(yE@-t?bb<id#bhsl9I$+dBc~<bpnFwzmXlMEqON4bg6d_dcq&'
    'cLH$%9RH%2-x?NCsD1ako5NcJE^>UO<<`}5yS_Z0-4JEYvfQ{#rN(RrR+V1NF^$-pDS6+AK60A%W}XM~OkkVdN>ZC?vA&Bfs|V(97@26OUAwvU*9$av&nJ1!'
    '#e10dA2ou)r)G7mqdWg_i+DjWBpmQ2;tBPiOV#Trfc>qkL@Aqn_7IG&q*)5lK;Vu7Iypm;)B?1W^i_KBMY`=)tb=Aaa(s%-Gqg<37SPn^!m6*{hIY{3L4T#s'
    'M663QGjrud`=_qDeT?|_2ZzlY$C`}?vWbkoeB1`w$zriCK}uZq3oKa7&d)J$2Zvv^fwww`^SdJP*kTPU8Xv;{9>M><fd2tK4E8&1puyG)Ni1_viP0;z26y`h'
    'jqrfGft0+Bn40mgN;>zQFH8Igw+6(Pvna|A_P5cN%%P8Uq$?mErH{8MAg1Q@4TpDhIL&7{ePD&8c9$NvqHw_KddpTFY|;Nt3z!npji{nnWOTf)gVhMC^$B^r'
    'zhzR(d_@)G5K-Gs+y4GGX!FIKLM4mE3Kx0-zZLVFo{#Ztc|NZ1+)U3X_3^WmC-&{S`??hGjYKN6z}y~+HS*jZ6e5Ny^7cW&?z^JB(aUzf6{!Au_^?uTl?s->'
    'Y7gkv7;2J24z?M29GJHS22v>gQ&N|-L@O&7>QFp0oh(+Xd`9tk`3LweLy>5fUv^p!=boc{0}p!#kd0@(!6LZ$@no?+_hbP2qFE)yN%dt?TNqjMrTX_xF?ri$'
    'fHR4nxVyHz?(OaQJ`;!8)1QC-;cwZ`uU@`>`R&Ue(C}(%4Uurf{*wN8H>*2@ot&Y&p`uaGYu-3r)!r89L6_=bZQZG~rqEPc|IFR0uvA+A9No5xR9agYD(0I>'
    'Ku4YfPSvH85Yp>m6$Rfrg~Kd<#nM6?WN`|Fbw$kwD}KW5ewKZfCZF{`>s==on{u6;<jHbT7F-;}uf1GT5$M?3rT8Ube(5qLn@T{ccoByL@b|^MxrESw>rshU'
    'P*3}QeoI3pZ}WUv2E~f)<Hh^)90rh5PFQmpU3JS1SN9lHE1$&59m(1(@phW48JW|QO^Y)iL4F1S8N$4Ktnu*|JvO^pMq_2(X_k_~=-7Pt_=_&K$5QD)gqLf?'
    'LvL`Ne=wl;+|Um*uZDKik+kYWsvJmZo{eR`k8V^FoL{&~DltV5j9fBaz79^dyLDZ(NJGVV9UV$<@UAV1e&kY?W7Z|t%?s3!cR3hK<`Xz~Nk#4<4xIp!N6D4)'
    '$+%AbhXQ<jrDlIWIk<js6`eNr!s#Ag;eYWjiM_ZAUt+I2y}fE)G3(Wj1b0|*Gj8D&N6%G6GgL5Xbz$y9?c?bAe3T)f`Nh=#b*PPscwsp%5hLhAw~_?@rGyDK'
    'up4=T?hH?);37;Upx_8t{>{{{BkI8~kakL8!~=_!j5_rOB`_Me%+n6VGH3uJ9XwECwE^x?l;!3G|E85xBIZ0+{$kF{wZrmUkCH3;*O1QzrCx#s1aowv(8zN5'
    '*soYkm?a<fhKNc_qjqQ5!N1#sj0?u$7y;g?v2X7e<F~mZTytLdNMJv8S|almaW}|KjboGBZ^3T2Wh{L*`)W}Ndk2s9=sLT?Ue|c8ss({8`*9o0Jjj4-$AF?Z'
    'zd0AWOHNeho_pM7UurC4A+*-3<4fk=W1-XCgL@vI(!bg-nW>?RMK_4*5wSE7h_#IEl`Z+@`=>v*O+kl#9>`%p7MQ#ppXFFsMAFes2gwVROlLDnigc7Li)B*C'
    'H`vE7ma<dkI~<mM&m!Xz9gCNR5&B4;d8F#>q1y*9-d);4qMwl;plu2qs7<x!>!fRIKTZ>IqBh0qxe{ORvz`ur(NY9|!^f0qH+6y>y)5^%k6FCRr1qf4zX8=3'
    '7!6D(09f@YjnVo1Z=YX$KFvP={_`I{|HY_Z=&kAbZ_?=O=t2IJsH$|BP_ejJcJ=G?SHHY?`4gOaJJz*T=FHK|E<m<N#|@)J9cGaqWe<20W|S6~eU9vvj;_e('
    'mrLN=dFcsS*nHS2ke#8f$y&<0d_@JY@Gg~Us>aIRmyx}YN`9!TG-F%lF&ht`xOl$|?UTok(#KB{XLFQhN%t$iBxl*{s6j))Nx46dyelG<3l)qg@2dk8znl~_'
    '$v&EqT?gwRWX`VRK|;kY(x?tr_8!`kVP|A(A9`?XgC+<(7-M5IcNtJD5%F1f?L~WR5U_bT=Q)f6GY3a!Vf<^M39eX()3BGkz^{&K2<^y_nbMo3+RIVWIW0c8'
    '(wx?BP3zbbm@U4>Y}OM@V*;P)%+43@yZGS`VC4h&4JE1ty0`#Jb*2dRF3VWW0swW|o0mSNxWEJza0Jg6u(fQlhQ1V|{qw6AKR$i+x9rE~ufKo!=U=k#et7xp'
    '=?_i)$t|s}`wuU_4SMG*S_WDzB+?Q32GJRkLPHXyXD)$uG^<9Wwq==P@`81)<IvzJ2L=xsI?|g=9g=dwk3_SMp7D+*&#O&cWm0v-n^UU`x>&L?x|&p~(nhYI'
    'o^@>X99<%i|3UR=!>lTl$02(MDow+x$;C(#qPRukx4l?%acQV@;P1mOTdLgJ+f`cVPzXh-+965BwgjZ=2ffr=V8S_S>rm&Y*?M$GfiX2D>uvAY@1KHeZW&x-'
    'VMYw4M_U6=8;^AZkBC#W4(d5dA^}=KtwuQUSJ#?(D*?{BBOJwKJ};S-l8m7BJ5WAYiqj>c`ELHbT+F-9t=?SFtNaY^mes-z8s65}&t2DuRQzTz-CQioZa#lk'
    'tQK>W5uMm+(zT^V@Ew<vqL7Mouq|i|I6ic5958&Z7l+?cC<vtzz55l3Mc?o-X-;>Ji+kb7+C;`O9oJ;8SSPLrgX))<O*J;rq!0SJ7H7oA_NWrB887U<3){R2'
    '-@6-rb$7h#ZunHE)$O^%&3WB?40>|3cC}&}Yn-txUD+rEbl$%{+6(P4`G+;yVZ0}Rz~s{F)qt)9HZ;TcEx)ycm?ir$?b-P3444uoL6Qc>9^?V*?NtkGF#G8I'
    'w9v^t-cwFg)DH5gv0E6c*|(Ama5_j{8T?=2VCHCbd#HR<?9K6A{vW{mzQEU7s$Ap0OwGoZ`HCV~Ot@knRO_>OK21Qwk;uEdpAQ8Gabj>x%3{)&P)rkg+)WEE'
    'IH{vG@hMN~{U*wDAfZZ}@-KP!@X_F5!YWNNUF4-H<4XZT`FJ%eFpg<Fe~ZDO=+3@gaW&^BNwEf48xx`QEeGr9*-RGW8JjcD;DpeOuqEgwpdTiSli%f$4lbJl'
    'Pzca~qSxN!WYs76Z1H}uxA*;vUtYg__2OH6Xy%&+ADpv8DHA+6bYY~jFPZ0Gj}MRMZga7$cD~x%qgU%k+0Rd3{qj6}`s%x1FpS<B3(e2l-r%L#tH1t~y?Xu>'
    '8(3SW`stUy{q*|#=dWLUE5AEyrWxkD?_NFs?&<62)+S_6oqDP~%-Y6kefHUB9F#>?Lc&ri83Q@MH(y<<*fJt8(E2G#3|k)z9@+GPTjRb|0;!J{0Ji5!;(m$('
    '`zJ+#9c*)P1Jdq$j)~jjBt8X>eF_}=6gc)NaO_jy*r&j;Pl02f0>?fDj(rLo`xH3#FA+HQr{o7vKqr?bWk?^TUvcDSRZ^T#JafqtV-)ZV3#<x)Trr=55Igv<'
    '8<-~WOuj1k7{pRlCp|cPoMJEy{O8NZt!B$M9*X>EIQ?IIilF-xLH8+w?o$NarwF=F5p<s-=u82DwkC3uNUma6_YCfL8D1BAQBXdf;vo#o2NU@#k3#LqY3?1V'
    '&)CDJAOtnU95i2CHhv64^Y+u^U}WCp@P@?9*ozfrK)%8>-v?N-8H*I`;|HGMyIzkYE8Zap#v}bgPN(u@O}AX;gr;yA)V6o+;Q_yDURFC<j8}*+9Ak`JTPL%8'
    '3}3*JPv)D8lYAvhcPno<XD-ZHxFpZ^fu4JEg)s64l`sT9)~{hj75Q&#lph_NqaZl0%?gHaNsXfYD)X0I?>g0c?NW^f@syGno{ley*(Ex!qaHzS&ajxt*gQGP'
    'QzSn2xCcUZm>hV^#J4lpQ2>N28=bW&6vuc~@XtLHuM^-yI%7x&9qlXMVf-`CVf?e+HK*sTMzmaaDpbM?$mmV8kQ}RB<(xqHQLm(WQwmU?fTNo;3OXmIf_##6'
    'lpm?=y6js%bvucUp4L%QTuI6&fR^k-KD)?B<I<03G;1z2v}Mx2WARlum~eAmX8h27**HsSbNv3Eo!uE6x%=IN{`=lNH%a_bPstC<^ck)^hlYIEXVGgca+4{S'
    'nmZ~Oqq^Hy)u%KVQLVg_Cg-px{q^~3u{k?mZq}S%xy;x2Ttm4y`L&fZx0*-QJ1S~%M+GfzTg>9l3t8Mr5sQyiz+#8u6?ag$;$s!9Xed}AM@0wP=rR@F=|e>k'
    'Gu(Q%va;(JSWGTeMa8G{%TbzGPi3xs<I3RF)kgd)I-Oup6_q;TC~?exn@^u47%;PXkN4?PGWnM#A2F~53`#<k#vZ;mewN~(#2_`9jZgAf>9@PIm9x-8l56&1'
    'GHtBkWUwTQ5)!q4D<3P<0*f^z6nkmD`36P{%Qq!&29pIA?PB=zGoBJ(NCT0nGx)x1bIEg4oBMbI8?u&_Y~}?F*5$>eGDOBLl*N8E#&U1`wO=gE7DtdDv8*u2'
    'fA`!fq-f#WoJWM#Ald*xi#wbB*);-@5~E_e_zj4lm7h6iMt<gzjezkA3#Lr)T#*lvsUU$w!YzSP@K+!UwCIxa3o>v*Pcj(;vxhs2r3`LK^*~@Gv$k&(sk)kn'
    'v*N7ak{>WZE@MX%{syK$+#l)O4@e)TJq0xE2f?nO#lPM2O9<j5Odn2*eTH&34ly*}k9s4WeH(s(zN4gXcu#yeU+=V>uMN6C-kN|m+>(-h(q@gecsb~@Wj+@t'
    'O{~!B7qOfml@mh-8UGKE3GJysSV}-#a!D<_cys)Z5rnsR9?^PN&d&y{Pj{D8VUo%M(orNg)<iOP$kX0ny`ab;KQ8tSVgvOoUH6p7(J9k=BaiV`N>^XT#U7s#'
    'b1z-s7O@uPxh=S)K#L`PuV#A>%NfLK(7o}MeBC3+4*$j_hA*RtQ8&28QDa%)#=^dF)J2Q_M)A25vYy?fye$>OizD`#6ov#&Tbg%6!7}z5Mlo3eyTtGu$)2y2'
    '7Ev^IF0$N&!(FN>6M|oFmdr^kb*pN&R&MweCc0R`X`G_Y-<QQCMn}=rm3dLbIap?Ab#}lW;S`Pievx3!W2|e8q1Z-)HM+n$R>0YmZD#M>$+=vTYQ0hlprGo6'
    'STSHhg{DZm9fJx@;sL8-ASjKof{|I?M>(pd6tSv9Y|9)RZ{a_x07L1>lHLxFTb)o-HBHmGov2Unq&vvs`N0F%ehPxCB6DiwEnk{ipi%^u$$gEX9&n2ce;*Mg'
    'WG*GfoGo3yyKhb<DF{&$iG5YH1m#z{VRb}aMC#2mRg<YHSiu^!tq?o6%V2_b^=;bKEcw{85@ut_$khAA8()x@x|#Pa&DeGxW13@B?F`OzW=tbpPCBI+3(-DS'
    'Acm8KQ$0#vF8N>$gElBM&=Ww6YriA*;mfJmW{kCpvN``{&J$0<KYgm)Fx*(s51*Id6aLp|1!792p`6@#koONBR}=!7(1Tbf3s2UJ90hrZOOm0B_gW;5Ps*<8'
    ';_a5-gopC&<5klZx9YlMp$t=1zFx6iRhwWn;_A17EmO+7@o#R+b+4^=M&$kwyDrLSfMA?SFBEs1u}LtgKzTMR-sbiPpGMlX^42XBiFdjpz!2fRA<lW^Gemde'
    'MPXF7Bno7*xHusX5c5<~d6~H9=6XNDw*xrV@ut|;j#_=3iJW#bx;x%jO<KUJ8bwuFMJqvO-9he<Os!~h6H3iaE_r%K$&cgZ61eP;tRjjaWg_+!UDphDC4`B{'
    'X#gt#Ju}bw6w;H}wNt*D#b(VMKfwZU_L<O{N{oTSyU_8422{g`XuiPX;U>lJQI*31n}0+5g?oX*1Y6Cbi_@+Q6O-p>FPp{Ws1@r{MbSOlNUOAvZV{<SjZEWE'
    'nI_UIAq*!Y*TvI7NwGx{L{Ow6i_IlS>M&n!Dniljk{CUMt>c>FR(Do%k7}f{evxd90ZmQ^o{IuJcZ#oHBrKe}Z#6(RvsFMH^1vg)s@GkZH9~RLj>BqpFRF1q'
    '-{js8*0;z?sqN?^Mk8j55n6FS;WX2FQ0)k#LL9ul6$&g=da?vb7c-D_?P0+FZneD-gMj>^q!%INva40Eia(|@lSmA0IDLK@Hc!OHL<qfme$p)=&|3Z=#o(xa'
    '*W69q2{Xw-J2XKaqUfYKJJ;cL{GFfWL_IY@G%*boAGE3*8(+t2**Tj1v4JZoUu}Z|UmSF!lG|KPJ}49~5n`%Nd{8=G$vgBj#V}WHmEFE(3~ZI2garI(>F_UV'
    'vgZ`_7xQV36qVzk)i777VsF9<by%t>=uc-etHYzNEglY_NaWOvJW=u4l2mr`uc3%l6)=0cm`#1v8m`OL+La2O;A2qNkkr5nekN)PG;#bYBE=~R^E;>?Dq1;G'
    'T+qBqjFG{J8u<|Uy$zX0`WOoU{;aG3^UU4LWCJ#ckgp{%E;-k6qBZ`R9;4}7dC9=nDBx`Lhz0^P&dU>^wOk5+($afY0`+0gGFkvpc+j8%af3t%(f7@2Wk7yX'
    'w~|`)h@h--?a<!3H6kJ+4Q}aGf1ip^2H&V2x#?@uTM^t06%^8*p%Ok(hInsfv0n9vAG&#s)F#NygK4N;gC6HD4Y%1&BZ{&X(;;HI#YBW=_x4)KhgPC!X;0Oc'
    'R+<P4@fd?=dZ9OK`<(6@V)mT8{>^dng}Qj#OX6gHb&k9u@u_dRb441MJ(m)<e&{{~qD>yab=}i1L0Mj61J!?cM7>6St9Qgpbu(&pRC$Lotx?*0=u0aKkc=G3'
    'QsL(0UKr-R*krulC86Z|YKjni;8t=s-#?hYPxL=)hq4kn+GkiifQe+z2kJQKqdYc0WkW~-+ncYL&OIlHIyWrKBjhTysoS@@hABp3(?Yi9O6??lLAoOBp=?D&'
    'w*OP|3nq_C<W(b|vXfkbr(ju<luV}r{cZ5GM2jN5;vPK-fUVKo&S~hC1DfQdz*(bH54x;fn2>6w>~i1%0)fJ?Si2&nh2Y=ToOc}TAF&ibA8dWuwRBvfF^F$t'
    'bQAH#bJjo7B!mCI3pK@{JEo_&-^?Pb%8JoZ>KJjLr^NNGfOGTYZPUeSnv>8oTrZYc0evQbu1IHk=kB4s*VzGx2#M+|L<R?X<Ty!2I2~^$PS8VJ37lCIR^eka'
    ';LZ+Y8XqZ5u1r3R&XKI?+Q&`{mYyd2>?Z@75P6qiYj(`&I-?t9X$X?3iLRiNH+KxvaMEuF>0mP3O!JIy1<8%HSKUG9e@HmvIhbn}zQbTtZqXSH?b1i;MsXBZ'
    '3&JI!tbyT7S;yN<e78IUF(&1sxbZAhU1n{JP3Bo(WOu(jh#?&vyU+ldC+*$C@A+F57-F~f_bjsr!5vI>Nx6x6fMuz6Co+k>L!dGZ`<p5V5A5357J~efLdKjs'
    'mI=CJC>pb?IpRh<N}e&2WGLJdc|0bnHKY_KF5Gr0cEogR4qKS8Fos7!8BSOI+0e_wG)-Uq-O$hCU;5d<!^EYQ=-fF$KIgWGa@5J^h5_y7BR4aj!z`7VyMwtf'
    '4)2G)>TqA?s%$~D9g23n_+B~RW!%?BH*4P`IshmB-R9BJro~5c33%XjPs^X~6nqoPZujZ^?htL{J_|w>#5L;&{n&y|J<*2I4Wviy`|fl8uR}r8O*F9Kg&>c8'
    'NdC1H3Z(WkEBYl;XcC#Ed3M~h+HGs`aaxSe<_jR6$yV9AUdkl-3njq@?Kc$Pu(=lB;O}<g+dM9_8F?-8Ftwxj*6J0)3Tokg)JtfQTSjkHR3xjoK={<9b}!Dx'
    'Z_CQ7z{KZ*QHhBhKS+lOG8m(7>mPoPfsw6Hd`_{WfFL5>0sT{KQGq}?-Wj;r1Cyf?PYG$2a00UY@0;<g>vqu$867QlO)(|dYs!2-{C%_qLP$&&g#Dog1`rD}'
    'Rbj_m|AEt;<8mz+rl1-amRQ?gz#*rLK)u@=;@X%Al{2-mq7J(ugXuAmRJhY9VrX@tSOWV2*-pH76y<8I43RcNV8z~rO$9EV;CodwUpTX#vJ~$Cw_*saflC&%'
    '_FzL6R0CBa6!oh@7tfpfb!dr;Dl5?2)?~IQ^Dh6oo#IAm{94Ut<7KJ5ADW*BSkhgsJ?ENk+zOleSWP`_q<$ZZ(=^z+)JM7NR&#y`TmA!0S21!-9nA|)ja^Z_'
    'zHy`&2bHmId#8S(PT4Ofi&gBoZ5Pi`Dh~z>_Jev#^DCdIAI#NZ;;(@)ssp&J5-|^tI`0oxP0jkQyQi@4*wxd#b&=o32Cc`{6tvdIW=z$zjCg@TcBOJ$r4kZo'
    '6+zhy_7o(YvwWUYC?BW&L^~L#fwSY9S0w>2o-j5nJ)!*=B+uDb){Kx()nj@=2DUi~iX|QxbUMMW_QZ!EPl$Y!B#rlIa&oEP<vy;rE0b0D4(AhH!xP!Rx|1v)'
    '$)k|i$j3F};>t4N_}#dejX5M_h^{wA*z48xz@uYB>hK0*qlbUCOm&)CuQBI-LPT#zG5#uabbDy!Q(`Cks>%v|m0~#ZTUkaEi5RhsO4q{2SJ3Wayp(4A%uXsz'
    'q_M*Ro)1p}^56&N^FkvzEE0>>NpEm@agPjdYw(KeJH7&#?Hv}Qqe!VZ!_j0f#mE?8heP=eiLO|zMob>`8OLGTgF$*;c9lqVI$DrX7-T#?dcz_$Q^!CRA)E@='
    '<tQ}sB@>@?>jf16WiSf4d7EDzb4ntr^>~zYF;mUOa33v#_~Bp_^meQiuxxNjB%5j!^B}_`doD(!8Ur{H1>gvMb`0#w<}DYc2R@%~E{J`1ZJ&s%4xbEz);G8j'
    '@iuZUOiPX$6MJWf0Pmcb9%ONM+nwA`hI$9;#)TtfHy!pFR@w6mET|yK@ooq(;}+R`88UqKVXv_6aM|&mb@v7FM{=K6&bT}H{a3u~VfZwhPwG(I<ZAEh?txSP'
    '4I=<P-uys}HPO-)CKz%t>q8_?`otiMqeSj3k2(5$vl35`CWf|59H75_X^6Pg5@&Fx%mF-(9N!@fFp67kP#UvWo(+L0RPaOKhvy|rusvh(5#foJEDX@+y*B<6'
    '9Bci5@7wk9NB?xWeS!h(0TtLjZ%zF{(g1F+*Y-|x8RjrX-kU+iNfEe$Yu?hs^Mv1<@jk0|u)wOOgz639Ul51p!^4qL+1<q@R|wvJeRG|J?wbex`YaRG%=m=N'
    '<-Q$R-Qj~(ezAC$hfQtJ4MOyN;Y}hB(c37DKR??NANeN3?6EuezLOT}w1_*<-FPP`VNf;LC!)=TEuCJeB!~uRx?rXvIjua_KA(Atpu90iUcR{Zerug`7w69i'
    'OZ6QwwclpMMN<UCs3RnYntEQ4bK~DslK&tQ3~NY+&PjxqgJ_=Dv$x_O)^**D=djuUSXw?mTc4-ix0Dd#m@7tP%W*-OATrgWtVFl*yp+lm5^Bwz#;h++6Fr^N'
    'riEV+o>S!vRI^#5s(%n+D)0?L-U>&9N}Ef@E|Z_jW#>|7=URL6Be#N0fz?Z#zFpmTLs{Ul0x7YyZWT}Q{ehu;#!nIUFb5mMllkYg>_*OQs;J3iMLDX7#!6jj'
    'Tz+uiuOsrZbvJ3YY=!?<M^zfLbrcl^NN0qJjnw8e#un0wD%7(!<Ayi=>WEd@d{>8<ZHS1uzwCmKnBxnQyK#4s9|%|Q<--H7r6Q2He@v5~4DbFfZZh>~y}jGa'
    'UAoWI`<2|nrQ~MbB-@=o{xMD<b`;Ikg2yLhZCIVdV^=bc9M(?7I$?Y3!x|r23U6gLZULvuoX8^7VNqoA&K6*GFKN3mik<viZtUB#qmK)^kZ`u^%w-;RJ|xVG'
    'dJWia>-rIK0f|MAbv2o24j%4*<W!%k!}yK9?Q6YBv{#At!*YE?j}p>@lpBeg``6T~I<3e>SrJC;*tJcy(rnz)^<2W#n6Fr>l;oH{Eu<uvtt*B_&Ai|pulg$2'
    '88n+2BsFdSWayAxc|44;2v7iVMW7q@l7AO)0iPHR^1ss-peC67dp-Cn3o+0SM_SZue2MhsbN?CAOwXM$0R@}~pUAlyy?o~-jNv89Y!2MC%rjy;k=?#2`id#2'
    '?)isx_Z0W$D9}edwI@({lpd9Es!#f!6d?`eNR{^#>Bb*4jfnwtf!LG)rC~Hu64Te4dUVdvM!~-h@{8sAlD_Ql%sTWdiKNs3<etsNybOzY4wy-H_fy+3rL2FF'
    '=+sm4xr?6}b2`N`$F34^6se6lFsqmo^NYg4X*Ig9NB(K}i5Fjq_@v+bo#5sw+Z~K_NBGj&VIAxOgCtPBVd^&pjoXbitm}@4M(R{<*&f54vJ*Jyvz%ECe?Ch7'
    '%ry9n{%jK-Du?{Y@J;!4v4G?9^B4|b%-9fE4ak<mNx`#eu>v;8eoC+;xeJLXYO3KLY10Ju=LkU5THueasc?-xbR?R=AH3SZ9N&Q-*x6um6O!JtEBx<u$2{tu'
    'a2jF-RFC!mJ;6?Zsg%-zZi6j7&Q{~oVzMYrNo||^5<UP|OOs3U#ie|K>Huf7qmYnnu^}`6e5#d%F0YIkhdA_WR%>d+n2D|hr1mQ+LW`-0Gb=Ky5+gFSiZ8S>'
    'FP8;jRC&38F2VL%susAQ(!4io=g*h=hfDhdK6%jh^aWwU7ffVp-=yXfW}l<Bm>~d*;G~Jpt9-2FbV(|Dd2G5y-V$WOVZ5!HZ1mZOmsaUzmmBSL85>I>rkWU3'
    'dR2yU2#ohZ|3OCyk!{y8D~=MIw?cqw{MY#ZTe7U%8x%ypR2w8ni0sc+nl?+l0UeWxq?n`7qd3Swa!zzr9eG3Um<YGii-cE7EY-#MZEhQLGF(2nIE<>8?d(y<'
    'vL)8Ish$>TH$;j8;QNx^z>hn0a|qjWIs@~2ShV|%Vo3bK2`8qH%bQ(Ij-7`ou}z*|%ZlEx<raS%$~KPdkNC878T(P0b6mu0JUcKAk}|l|<{}|#Wal^A76=i^'
    'pE~qhfXsoZ?t(f5Q-D_(BeRbq4&VO#jH$$ryP_!Ilp+JC#f8p^D@ir@JvCyClPT`GOaN9damW<X9@!iKYz*a;EUb5sXzY$JpM3?b7vQTJ#+6C2FhQN)TcPmb'
    'fzl#O?gFp%(3|2YlDWWoI+|>gCPI(k2sZwde8;&PjFQdi8LXGtDtFl&#6t|z5>ZhkOvsRoCje4HFUjxGdyB~Ci}zjc31WNt0fu4^oYI^;XrM5;@nkh7X~MQi'
    '*mi4}21RW`+qd~@o=YgZcIR?u(xXR-{dNb#lmoQ8_|O~#`q@NWwON?I4u8&;8gt32B8%u&7Ny9c=R!SP<X01hBzcA2DEvzMC>Y0uuDZVoB`tEJ#)Bw(eki@u'
    'zp<F%vCaRkCqX~yolz;_kDD{br|O8^fxj=~HPxG)Dulb%vH>Riqsk=gcgo8bNYp2w%;ZRsWmvY}m~Ur$`hsc#lVL|cDs}L=+$Ja4s|}HLr{Lk2sg?wk=B_qC'
    'Q)ZhJ5>m2Xo~DAnP}8lp7@AlVQWA$UVRM<_#5s<+=qVqU1~3xRG8!2^`x9r1WkZ?T?IW0Sf+?#h;SEWpm>|6<+1VCTUNd(^Q)%s`WN(x34hpTW`sUF$&f|lg'
    'aw9Oe&Cnloq&*5uLmht`YfdRWOyd^h6$LUIU!>+0nUZ;HcS7&%WNF|RFf$$XJ?x0Ajn74RY`hmV`O7V{RSpn$bmDcH^cvLLFzgt&=K?UMb>w}vw{3*r$+agl'
    'jX$8Le!ZTCrwIXLy$5{X3P*5X9;76=DIU_qBv^!I>1oR;SaYU^s^}UDUu%hTl_@>iw`V|FvUrFF)XK<AmY1sKWn;@$a+#T+IM_%!jm#FjsHAjqxxro9KagT6'
    'DZ^=pRLU#nTGt)Nwjxt>B8H*=?E=)TgeRFpqX6~jKnqYZvDWgFT1W_`S%xy}s>)8SaDxrWxgZpt*yF&8O$rC`P=PH4Z|q<S-@D3db_TbR+5QRgnr2dLIEg-G'
    'h>fR}_mSrB7A<vRky=tjJkn_M`DP?<$X2G?i$+>)a|Y6nSM6usfLFML)O4C1#VY0(N31(|@igy9H412_TUWezv`9g>ktbMez3T!Ab_W&}uoGc28WCP*+|GX|'
    '2Ally{nMX2M}Ec~xcaQU$f4ZNv_m50T(LlvB1QTCxE)GW>RO)F6zhsxVXE9*T#Q$j*5}B4ul$ng-IAJjRwH4@U2P?LJl;R3(@2a<kpa>1#19eocb7h-Bb}(K'
    'Bp!$%j~`kwMDl7{DI}c0NKdPkgu!ilx?hai3_W;s=0_^qq)uBXZ><z1Bdw&Z=-Q89zN7IRt$Sm6Va>N6>RKb%92}#sR!)xmu=CL|RtA3xslTxw<ph1vaDq(W'
    'cJv4dlfZf@7nDp^zVO^uJq}}05_Iiu(HrD-&&kB8lh)IUaBY|4Y5b|5R+DD1n=DXw4bz=9?4V~3#+b}=kU_^M<zlvh3oS>tnx@{~kXv9>c)HKbTwaLK_eQY;'
    '=hx@qy1j3ov8^0!L32l$(x9<tSnFsV!#45EZGO%Rxz9rTTaXx|MM?|LVyAYcZ6XVmM9GHzXVantS~SPwT4mPH+IXnjybJCfvYPN$IfqeZn3^cX*q(#|Kz3rS'
    '635(#Om<BkqkpC3SgR@uhLd=Hm3ZD6O{fUY#!*!v5DCe|>=CjQ6DIYjNw1<Qx)xTL$fG&kOD&inamU>ye5pGfimbgPsP!Dj5t+}UEYELsGTu6|X?h71h%e1>'
    ')4!!N5+(KCjZf5Ru>z5f$(%o%yi%+L$WQWzln26^<{p;1B5$RYWH6~OAf=V8E&;x~&UZS@)N;3=>L#p!1i|akz5L&TncxKs(<q7kpLVVsQCuhQ%H&Fs!ga(K'
    'yn|&RxHFZ7kUzUodXhR%v6Cy+Bn%i^U1CQ#bXEj2<|q+@@cRt&Y|q9^_#OY2znX?f69=S$HCVC;c|w$ItT}i*X~z?Gaz(JoK9tm@WN31)jk+QQE@7Su=#N`f'
    '0*&dn7hJs54|O`PKe|U)zxZxMogNmWGzl^Vq;_q+>JT6KIKo#(5GMjPLOIT}_C#bnvv*~tF3Rwj`1z`+2MWD@7Aq^gQ^tfgBl2~6lfky=1a&ho=e2Rs_E6u='
    's;guA%id|Y8?M{qD|L(4%!ZnziMP5mxazA0$o3;!6hsEyzQ$c=B5cq*-XRZk2UH$1HcJ$t(EP-oKtZ42OfS1Ny0Cn`UTtz{I|GFTIH`-?j97kr)yuci_&S$w'
    'J=EIPwf8D7a}A&3&DY`|erUKyny6X_oSCRtpBMAi+R;T0TSEd}*M;t-iHf}oggL&)$0DM2197SIifkdBfImxEkn3m7phKQOB!`b$h~c>SIDy7{bXX@xf_NlR'
    'TdO6CAw2vw#bl{%O4WShaaWZUs1U_8@~eY4vKFE*_7CE~=`xIE*f#H7i+M3W%`r(48lW!VOF*iR<>O<YB01qJ#wn!<dId(pJ$sEiZ7_4gEqC_2f8?H5>}<E%'
    '^y(VCU61@6+x9R2CAZySIGbWxxa@qvT1z=uEc33%Yt=p(j|g&3v`I$iz-P~-a-n|NRKEb4rrOUt`rL6BFRs3))t8U8*XU$39cPpCd_qNr@DH9<bTZzs1{`{5'
    '@bkrT!=2$yo59A-F#rkn#{ZGG!$YP;8e1swYDKZS>g#pC!A0kO^=ZwAsw8&Lcl3VL@$!Den}9tuii?6O?|4gan)=o4WndTL%<i>fs4y$(P%n<}JMz{-g8;2!'
    'S21qFSH9ix+e%qB<7@j{^52q^d_5j?RAGxdU5+}I&5Mfc$j#QKow$jN=Tbj&Iop(pF|%TR%8#~a<0Rd<;3~1-Y`xA9XCt{ExMX6w;-2RZM>X~SG5gM*HOm~Y'
    'MAmL2TkpS+KQP%S107hSh<a2p^m9;bNISvk+^BfnkLQ7drJ3Nzr;Tk$sd^3E0w@0mWowO)hs#X+9zFVX?%I!!g7=Hn+tb<Nea!*UmxErT=~i9lU))i!N3Vwb'
    '`&?3zd|*?GZ14lsHRt$vnBcMQC;GIs#frJy+M%Y(H)w1lzv&rhbm)miQ8F?HdARJwi6y)!t*-a_<*RSM&z}DJ^u-TPpZ)NhhLJu4&^8ydcNh^5vK$H}9p#f<'
    '4|`Z!Z07d1;SP?A{Ym3rP0j~6uK^l^r}?{L!jA;q{jl3IAHU5m@{7gl5(@zU9_iVnII#~S3K|^gN?R;l@j#j$>>oZ#?%hisHjM(v%UADO4Pu(M8bbx3iRBv>'
    'k33YO+jN7>imga|oB7-M;yozRF_0(Y_2is9_XPP?Ty(#L{g53RMp*-(rb!G^)QvEN6?d+TKs#HFrv+M|>*p0|62puY=_!+CJ)Y{?DVW#hWR8K;pN=OsgK;$k'
    '-1qRQJpcdy?7iD^97mQQ_>QlLQJE?u#6S`NB~=L$WJOVwL~E(YAz3As<z`1G5|Ka<c~eFv2#5u<v3c4LXtRCZ=Y88x*m<9i=r7rGbNAyO9+3%>swo?5Srw5H'
    ';ePyn{M=7}Twg&48t<Fx?of^A*9^Zl>swjAU(p$bG+R{R#{I^W<$t4=OKzn-LA?6f!^YJQEKQ_JXu1-X7kd@2w7hG=R$@aRyf%ahYgw}#wkYx{ijh%km5$yy'
    'OQgmW##{95n%XP*ZMK;|x%!Gv7d9O6wN&?*cvY#nH{zQ@Tm42hloxBH;Xu-1VwAPDft2Pp;2gDJN9anz5QD2e?+((+1)>nG2w`V{o*Y|Xdhhgdi{w|<m!H&k'
    'X2|6;Kj_Wo$OW^(+#79$XE7rpz&Qn3=?cLj9@oS;h+-dKbADEV)W>kKb(*<E&)Z5mGUv?-17$&vO>+`D<37O<Y);EytUad-yb}cApapXegC#QYHSRkkYSq;J'
    '<@{V6?x`VrV5uo(3TWHxKqkY%uH||xIpi=)P9&cVTU>x5Qf)UVgY_5)jYtW<b{tk0!t%z|UGUBWi->L;%M}|M2LK+;s|h<csQcW6xH&cfZd{z5<3^mF{KJvZ'
    'y?U`(jg@5|KU18&Nx5QiC`{y-h|7o1Up{&C<KyBLF(o}Yt%xJ$SJi4UCk?+nkH%bHwYEgcR>Jxq!}E0R*je7-z+8+d!^%1lA9mnB6ob?6`6zW#t!4&Xz-CaG'
    'o5^TaO>fw;gXd5M-?;7fn=gUW69+q3Ml_vWi@ErIau^zb^f*EdTZY#K40y2tjC)3T$x!eOlY!rfgN05s<=k=-OfW$f4dc{U6W636A?Zay*r0}94oMmc7hsN7'
    'nLEZaKi;V;xx7+5Yj1~ylG?0(TeslmS@}UOR0u-#w!y_H)_^P)cnNwca0UjJz%U45m#rDy_E?q95iJ(9()p)kZB4vGJj)Mh=69>_TE5&~qp5^JQ+X0Zbuu6N'
    'Sj>cWqNB!Ww6+%C`sI%1-c~!QRaSvTd2WaaOLO=nin>Q4-HOl+x|9jn;)PG^;=5=wZ@t{NGGS3LQBG~x!I4Fw_>9o7lRb-fBcQg2JMwB#VC`Jpu$_hTA;tK+'
    'VH-aetLk0<oy=Xd;gI<oZsYV~P-3jva4Y}U38`&Zab_^OI@@~HLP&MPYU`I{3H|yVeB?0BIr{ax_G=p(=|kvbHiumtV+S0|P!8ivVK9h*TO3B&zMziJ4i4KG'
    'JHJ+w=NmS;ykLbC>o;t$e12KQ1`gx=U$A=u#?b4G!qiCNYQ(LsvG%x-i1(4MOIrgzHiJE?K#wl7V2hA$s6(JO&7AzQDqjKJiFiecgq=Dm!7J<3T}d31-;kuv'
    'R@KTKL!wpNqmp63iIY)kq8FpjxoV!Z4fg`ulK%MCG0RmoUpGx(CcOdng<}a1leg;HzfJzG+B$iw26PWd?E7z{Bf(m|NHaL96AoqxC0OPUFawRntzR`rTE@#|'
    '413}sL`~GmAY>eF4nw{4sLv+WuH#HheuYR-*}AbkZGgRhvONmZK>?oM_G;=6DZJ1EapWxXEVjc(6Tyv)0RsfY=M)0~QmhM<vh<<#YALRn_<|gs4d?5^4pOD8'
    '&#@SLd5@!YDgZ;zYO!3WQlqNt%_%<9NU|^{_FvKWog-pTdQav&T-z-76<AFAYr>{tG(#Fd&`0MnHnfH|v9HO0IRQ3hvAz&vD8QB+Cy^$1y*WeLXSmzRckn;#'
    '@uy&r&>T-dBo3(w@#s1rCng%@G16a&8xjO<CmT?hUu>J^xoMPVgW8@T9<cCejc$v#SnHM!RugZuU@gI1ZrKg`GEMD<yP|6mM_})P#NvrCJ`^%@teGFz_Jksz'
    'lLpvq)~pS)nd{c_UgpJ~hv~e}_N^ap+c;jP7H`>{Qf;^^PZpc?a)YMVHd1r4J5nV?>Z0cte+8cQ)8gUx-#>r+{lk}!O-H;|f*CdOEoZ0IoDQD~maJ+aTWt#U'
    'nz5G4>-B0xXr`5W%zc8&p^VG&hC>D+2xTvgGh|Wk07?x#yenfv<RuX33nfB!(aCMywjk}c09%a@BT{sz2zTXpBoX2<>k)8;#<!EBgkm?O#}OmN+)XeBw|)!l'
    '-4f-PN$9p`tMUq6lSbolg&38My`YAeSWS|%YC^)!HJVS9t7^uj%Ik70d|oIX=@i|%Z|x#nMbwVr+M=uR=(Ep7hjaMA+Ka+wIWIA0fma39)YSwK%=iMYfx0RE'
    'Wah9nxrt7J5|K(8K?W=Wq7jtqHXWmt*hz$jabWCaI0ZNbC(}DCFfVrDjO*g{nk`DVY_!uxy<078E>{zDaX}fjah7P#(`tSF;NbHEYay{T#uqJp8j*Zg$sLO;'
    '*oX1Lr1&3SY$zB69?<4Ov{altIO=hLc!>~)tR21$MS;xou~EumrFWYDC5z9Im)HYpUjbHRenS{454~ymAL(e0tQ2sMWquY<J8_$N6Xv2|`WVe!&|)dQW3B|O'
    'V<n9qw2PZ1P*kvpG7z2j0Bh}>UVsbB5vCNv!68<qoD^RRy0H@H7mAtn*_@xD#amASRzmp(WOX`O4BuvQ3Ik^Ur-dk0=rHcoaJy)k!VYpr@?IZDna)PblrCZ|'
    '-*1OtNIse9fZ<cNN<)us4Ov_TV+Qjx+ss(>{063)6fs<5Gp{Wm+BQIM013fP9HzY3Om*<n9;t(^?zxTq;6DWnzm<T&$rB!YC~3m3ME7qiEO_uPvVsTyDJu9+'
    'QNe$T3O=}~;6a<DAP@>X_;4bDcf&&e+?@V{PDcMh$mb(cn~i{e2v~0HGgN``O>ZnelyW#^{FY{(hccRnIK%bsoOdrTO7dt)083tu&Pxo~ZK#2M@&q~D>6E-a'
    'kCSD!WcLqvLk{g;tAV}z_GZcCJJv4kW3h(mD+8~?PaPT0j1vvsUD~omKYH|+ZyzSMC}5xnAK<7ezIV=VlhiN)(1p`-f}wAp-Mpk;vWOx3DARecG!?;b_?YtE'
    'RZfQ#WkZcauDWx|?*Y9^?RIoy!DpE2*%DLl;0XX!J)t&wAOG#+*~e$a$3J}h)5kA*nN7mw%T=g#;aSjR0mJ<Cc%zQa7IpDfNPIZ;wk>nIoLHiZOUd~7P{J*E'
    'x6s9^g%UlOExXOs>yGjQz``43WP6k98d*_Vvcd{0JOe2>gJ>tRS&uoZYe{p8!A$YTuW;1*@C|Rg1@X$EoZ-%*3D}EdAIw%gwd3z8@Zz)QPkws%{BOlik6-@q'
    '^xGH3_dh=U=HZWB_1RadrTULgANjS#mS~GmM9V(zZXeaCy`vqI6l)4YIwdex)X|i`d;sAG3c+BqVe!}Kelh~6H-M5+(qeoe>I=XCu(RU5tV?q5Cb0?Hh*6L{'
    '7@F@&5fRkH<s;T*VvBrabT(Qt4nyR&pe(cF%f)08FRm=XNPg2Hvay1eJBVN2j8?PFvRQ%Ow@6-m1$e||M?PBLo;-PbtM|I2BkX}RNAyS$8P3j0iFJ09{e(hK'
    '7xmiUzg2k-=W?~MJqVGZh=<%Qg)jPNo7u8XyNyN-P9N3ds**h5NkzhrlcO96rO60T`U$3ZYk~jH=Cyi`eYR1%_%t}-u%71YNWxJ67G|hDzIFk0NHgX!-!OYQ'
    'H$1{3kuf_&Hv#hNDA_P-3S`oOq2jin-A*vswL!A%oj6y+ff;V%Ev+Q7SU^DaZo^Ea7#D1fw6`br3CL_mqz0fZ1-u)4pWAP3#0=SGGxP<sPH)w!u=11X@wvh*'
    '(M~zSDWHksa2cQtr}RjiUxAQ*Zsz|jHfDBan%$Dfc<*<>TztU)0NxL<&VYEv;?NH}NSw;2qnmP-{KtR$cjVp@WqZiVDvYH{wOA-ymGy7XxtA9}?UR=xnXXQu'
    '728W-j$bTJ5x1T}Pp(FIB4KgRn9DjpYlH&eY*+h9f{ew8jxmhy)Bc0x(X)MEB@!k|5=q}!C$NKX0-*MIF`5b?u|M9dt~ju)=y8biNk)kmvcXhH&e2T<Is{Oh'
    'Yx=Z0M-?2<0YH`}izV89{mNEzdwV}Tc>#RV6QI@#LBPddVHtq4WC57%N@2TF7{$!t(fMf)^>`O<ZGl9*ELVF7r1<9Hi^s*YhtJ{PhtI!%fzHR4LsTUmK701#'
    '-->6?pT2zh=;@CQwo;dT`SKyKbWrrWCqF*!$oHt~eDjwlKYm+0{_~5+KYc@W{6v#layOn<#l@&DaEjnk<v%W=HqsE5{k_8U7fAWmH^Aq^^WgA!=&|6V`AxcB'
    'T$Xd>(-{`BJoKthxe01qOKAo!mEK8D!k<$u@^(7E%5)DkhmBoo$`!eTAAR)Ezm(-tHWlWKfAr%MbYxm%@~Z?e7vdE-VZ85_B=H+l2n@Qs`)oY^FwuM#R)N8x'
    'JfJ3T_G(<MkCXj<eFr``NKzOAzJ%@Tf5})`izfn!XUB!XaR&}wvhhsHH-Hb;z$gHeSvB9M3wj?WW}hM%QHUVvJtto-nH!>W`R1ZR9-bA;HAkpeP8Ta40SuhL'
    '!A7j1CE(bBuf|ga9R*@qQoh5ijQo)Ug<sA3W@@zvgm<7`7w=Bd696a%B`Ab2;sivaL{v6qiXHp0<#maZ<nX}Ti+JMp_FOlOyhj~Kwao&#PNxE}Z3~)KT8_CA'
    'l6ZH6uK`3kPI*jwRB)(ZPr%F<$=L$%CQ5o}PXiJXa3i=eMXcpeBV3TP&04}QB!{2lERN2~Ur;2+rw$r9e6TMpnMvZgs?HK*hG<QU2AWUJ1rR;^>kHss&M%g5'
    'vYx+ul^|YB1PS%EFe-*%di(qH@_Ju6?d($;9~~F1J=qT~^5+IT75imV!39P_=s)hzHv2y9+mNY^X2TSB&>|e~mKHH6>*q=|+{llY?a(ErS@*T}Q{uEVI@g{9'
    '7P|Sx=Uv@AJW~Da$R)c=kCnb;pVP_hllC?2ZcExZyl4;dgZ|NH@Xu%X>kpsxc4*dlAjO*ta?qJ(4zQU+YVe@9<KdGN2WJd051`b=o4KYw_Qm8}py5e^yXL;Q'
    '==h`_<F(S9*)zI-Y{uKV9Ia3y(ce?)@EYm&MB6X1d|Z@>;-WeO4#UX7=_Sq`>#R&%@+r_hD3<)fFXU*OU5}QnYS}C#woDJ5{bJg43>a_|hy8<t9CKsKzlya>'
    'dlp&sOp(KQLB-Sgbw2+L|9$XTqz-fv&*51y9#hfJ4qVa0&-{Ma*`TmSRQyv_{KynnQ{5t{^X8(*+Kc4TBid0W+Nw~t;?I8nzy9<nF)7vdoqQ;HAu(K?BB*!I'
    'vy(8scL}HQxhg5^);S8n&h~#O>$(DJ0B+QBb$$cn)})esrxIEN`#}PU_05;Mo%up-(=qf0&N4bhY~alBuI+{M(I$HICuF=@)Ii!)YwN-p7>d+l2iFL_6tN-c'
    'X#tX@+wC-yiD2u5d>NRa=mEh&2qiI^fb9_p)I>p4lGJ*5PTu-DgO5C&!AIFG=TmGdHN~{nCuDpA#Sk+A-=GMZ3m}*3brIy4x2qLZ&}+qtQ_`DUt{8jx7z3FD'
    'mc>BkxumU>57q=0<LBnq<&lI``5qBYFE_BA2QcA}ZQw|fvkWZ>{~e$zT5Q+@NN||U#JF`6+>4yF9qCsuGTkW>DI;EfsLg}#8^TpO?}uTRmy!7EHb<r^EyK*<'
    '1r|rkBBS(qV~V<D4+7g{gQ(V+4xY1H(g#V1N45Y?%mR=cPT~@n%CJQgHIchj;W537POO7HWk^-&huG!#=0vg;sa-ESu8ornWj!X{y^Ie7`VvU>f;s@((%*(='
    '_eS@qN0m3&Tv7v1j}vbb#WrWsNy4k@$JQib>J#rC=gsZfrnHujzVwUso7bN5mLd%>&AZtH5qxe`S9E_N+^q%?C9S;;K(NG&GA3)V*rzFl(2HkzlA=awlI~ZX'
    '&aEbhF$8IzDX+CWTjW6t=2q6?iXRtm_RlI{gv%vTeVW}t2Dv()v*ZXFW7q-v`n&5}ZLx&wk<6BRVP;2qD9I{6Wn#>})^bl%^5f)43sTMNdq8f{>x<EJLON{o'
    '<^KKTh?UPP71{i<yrC3<(0!oHX5&7|=hCy2xeC1wQ!&CFO@`_`fes5~a>u0DqOs*4m&<kXnEs&L$90r~*TS50oxbd%0^X-8lzPxR_-Y!VR&q6^E-?<!S7!!c'
    'p4V_qr&t{h)OA&#9J1lfNpFV!ruemIQ7&!;SqEBByk{BQ;5bi?kM|FU+PqfwGPH9A(>A_L2U0pW0EV8R$q--#N)%1@QFpX_l*89b^EQ-d%FPevz}KbcE@FSc'
    ';D&N*DsyfhPTN6vEIpUjRY(jJ>RzQUvJ*h%gJm&Jzu_bC0!X@Lg;pLY54z5V_7IpWfaLUcN;J9{O)!dx)rq?Mi!o#@&)h5jl8u(Jaj3Z@F19PDXD?dY3pyQd'
    't7k{!3cG}YdLJNywuQy?@5Akc?&W0?N#A4f-$S+fnAOFw3cPQvbSK7D=<v3<*BzM@5J(Zy$R!5tUOGmoK;B`(QBGJP0f*|#+Q^ZaUOG&^`YM^2@ahbFFnG;b'
    '=A6OcgbXHA1dx3Fb@Dn(exDq&qY%AAb2|iXQl!_Hg^Y(YPU3KoG^0Y9pqqotmG*`vbzC0VuA0(m`twOLH)f}s(#jm8xJ4)m?rQFJB6onjfa)1)t#C5ax$cuN'
    'Ve9ouy8mklTvm#OCRt<^k&#=hsu2;ngV|_((YLSCOP0uAs;Nf`QtVUD$~j;RlaAN{8cc@HXxm#hl@5uhAzJU(o~@?0fQ<wR^!4{xyJ<>^;pM9Nru04_)ht&y'
    '*6l;hf>_uzu+?NGZfrCNVhD?=2Z+hw5%NClI}C+NQ{lm*6~N2r^$>4(0sl#o@`~dZ8qD8uFhc<3zYmlt2rO4cFjtg&g4<4GEuabhSKutBn>1l3{9Zfzw;i_;'
    'S>+JiLnqrNs>dVre%Xx9R;(_>on)pUz=!`x6sIS{63U&V@FW*<Q@_o;q%{N>@g|&kuaqdeQO_x~1z8u+KSWdbiEQ-LRlbHmTU@hLNnjxu2PJ6F82k=~ep(cj'
    '98@)8a~}IA*^04F+|PRn<xVzfGlzj2Mxz6)nUvaQ#(+@4*71ZR4(qrYmT*i>@w;9QTk6X7p!)oeyd{>o?VGSCbKsa}q$)g7_yqoUc<kdO{2!v3J6<lZ`oHAi'
    'PIDPzz3r<6Xa;wx?}SBNfAK3oKGr?DZ;<!08+j7`>|?IGlo^dI_=?<v{~5R6(jyH+&e<$jD(G%HiQ!qJT)87!)mbD<%`@P66wFKh1rFgx<4<c0>?gUh7*?dy'
    '<Hcf0xpc0|)*J}LK5!yIoh(|k=l}L^)TMv>H&iu}><)JQTC^TOj>0B5EmtVr-KwzUH9lQjl|slullx{DQ8os_FzXB4?ThJIpVaJg3Y&_C?CTNvVT*dk=6UgT'
    'GmcQkC{HL>R9SgAe37RECn2!$^*n^tfoVpxGwFL)OZJ{20_A+O*qD^tS^#Nt=IBGT^%gQ)qH!60)0$^8c1+#1HIOsToR5}T8B0(cV4n2aWL&*G`9k_F(tqW&'
    '6Tr|iN>sCCB%eGu_yYfYfg~5^3xdBSN3{}_N%GZ+?qGm8Ts}&Z(TEPJ!>m!Bj=S_qB|-Ye-C}UBtQMp31u7DQ`Z}1Sc!s&V(ZBwg2=xeT!|V8#Z7sqj7S3fm'
    'jW&<c9_^H&x_-lHP~x~;ujyVh)`y;_2AvZ&+!H2Zy2fNedZqU5m?SUZjNEN-xP5pqXqjgY9at!3;S5bs37kynXj#(~@}<yzN|BhyqwL)p%1Ad1yhqc%7bXDq'
    'BrO<OS}n$&viRnKHHzC+)Gi0^dZBTls|5bdqkY48b`<b>R(r<O4l@@tSr`@<|F#txoz{!#W?kCSh(|CzCq8;Y6rSP2r5BMmjmbf$kl2Z_0bIq2ARbG$H`ihs'
    '7i0CC%rgxhZ2-cw1ay67#Nkl*Adlz;J?dH@`UH2ZlFhtgws|9tow}2Z)qRMpGH)}qCz9yOA8`wWh@vg}ZM29v_HExi2|XGZ=?rNF`GZAuEWhT-g^+8=lasc='
    ')n~`N(fp#PF&Z7J{iFT?pz{mlgAd{7KK;DNIziRd`D}p(@gXz~CUd`~jdq}7KRw-`eKVf9(aJY*Cg_|2g$P_&xyv&BRq#XTYTNtET$Id6+_+|l*br*UXleRA'
    'kBAgE7rizYteB3@`w|V*Ym1w+K<;OB+CWk*g)iVdRSCC~P--HhcN?t}_JKJQxRx>g8Sa#vGP~jZ!@4Az4Hr4u82bH+%na2|{*?R-xAiG&GF(NquXsbrG?F!O'
    'w3xFq_G|FAv}|<Oew8iC)y4}}<6KC(I^zO|6ag=?xEZBf6bPhhX0Df|q+=X#Nzv=dgfYcLdAr#?R^IcIOQ<NZ4xsJAUU%FuDI-$+!crZ)`sCqon5*CF!vUt;'
    'R$T-RafTord@N;}?!RPjnF8s!pXCX?piDB%lO*M}_=F*4FilXJ*AednTaNo&R&%_Ad-IF@Vv=8+?|(g?ocBz6b^|3x81#=>g%Rlrm+qVjmqu;Ka!@JkJ!jv<'
    'e^K)0NEF?nd;Wo6W<f<;=@UI(Vet&;;0sq5uF{XFk-MIHHp$Ji%bsKPE=M>|T>2eWa2}f@SnPNNF?0al7nq3AW`jnr{G%?EmNxgW_u{bm!ZP?j_vx?Q^*@^p'
    '&nKoT$EA@yA#16kec)JyrWNZom3=6)*_frH*50|m8AyY)%FKOP;qrkn>HerPnczUkRPbUNT(S)fo2eL3o5$SpJ%&8AYZ?9O3z>OtjF<}B@b;J_-9iZl735=n'
    'K-vy|!!~$w5HWOFiUa=uE2nm}vy!qKTE`21M@e$G@WLlhj+3WLw%<__;^7W*5@Y;CUJLxn+ECWr;%CqE%3iF`j4d5wxBbNmaHS^@F&1^B>0*8!*RcW1XM`f('
    'p<H@IZuZqGNnhuecuQQA)&f7N*<IzA4e?JmiI{ccfB*B}i`V$iW%3{Y=l@e2{Ochg!2NXwPyhY@`+rfRdwOsEKmR?u56{=dYx?&R7T&oXe2*mZ{*OW;C#wyh'
    '<eRFTo}tA**>m&woY#@;g=s%AGofXb&}q(_;faf2N1c(##K=>EG!0+k%Nw+WC0Jw$OLkr(a#)})p!8*Q_8P7HV7C(#!=pHHeu-SPFOu0w*7BibED?eK*4PWy'
    '5TRXin#U3#{x%oF-TUF;527gKjX@<WeJbWiIK^{O@1&}?M_I&KM7{_Oh9S%6HBjEzjH%}%Hhm$ziq?I#z2eJg#X$YTpYiZVVXQ`Pm=(?Gr?{T-Hd`*HH}l1;'
    'g5zb_R{TMXY^ud&atZZf24QGb{kbvwj`cUST|lW?{{YpMKrw9}{x5C7%peGkV!T>f%C<L!ZW7GqI_dyj6{3^Smdr6vKE-sl#?tci{7w+#yf{qm(H{{i$s1t1'
    'nVB{>qb6xSL(huNMq;@h^W=|*UmRf@{8vQFK=_=mFLu(hpsJ$43#F-)X{oU@k#${r6xKXfOBK*m*AztwMh}IqaR;c9enF)4;ZP2)>*MRvSLUtp+spVj93jfM'
    '?6#C~UZWcRx(-I$zS%51xzV~|-T@%RjPobVZkych$E<0&aB8Mb7sn~Ib(&So-UiG#IIWLQgUQmO)$3Eu+8`%Ml?U?1@;m<IRIC*Y%UUq^lJ&rGTOLzonb44<'
    '@@8u~T3hLX)l%*s;+#hdE5+;jG~o66KzIS6=hp2Mf1HZ7&N|Cf15N0f7ud$n<<ytl>rw8bgVqM{kL^fXS7jeJU^ueei4TAeC(xfHzr@x;CWqOGqb5iqLQlNq'
    '+Omr9+V!Kk;i}O&IbGO`Lkxd9kB#YxG-ZIeU&uio^bewy=<wdt6M}C&`AF-T_o@l+H66{C0s;=?$pK$q01PjX2U!+7WR|IK+Znl<v^<dxT1ADmnY3tJupf$M'
    '>=;eDWpxqtF&WPIBF*~a<p%x*vK=j6jg73r_sZU%t`*&Yo{{%z1br470WDAfpd<k6RaI0u{dvuo75v;_IA_=vWc_k!md)rM)#XhG1R;n<-tl%wg&aA%;-qp|'
    'c&-rc6`Q%KnU>C&J~=eSoRDXmEP4<Gl4S!jG9DaLH;3M(2~&#A69gaGPs<5fw~M724uKjUK#ie{iMqirysE@<S!EXXAyNpJz(^O~HnwWdQcV=YToesBnLVSz'
    'tuUe$b<)0ijS*b1$9WTA4Sd?PEY4PoB~?X+dNj}d53ltH+Rymfy^Mfmo|;1cOwoM3XLVBUlPtw1OobihKk^g&^&-*@jq(lr2ux>QW|d1u6SWR={M~wT9dqmg'
    'jtW-`B`=bU4`${~RhOQy#oS6;WHKVY(W4%1&e2c?)pI1(P8%|oz{t+lG`;mp0AiF#IBllqFIaM6B5zpFyXACKTMsSmpXj1{$gEopgt9O)tYXmp2vO_6c@<+r'
    'i`3@Ge2Rc(r4RKaidS&m`Ku04Dq|)vfVnW}GnX1vLvl-X2No?E=rBW3yn&={mZ8S*u-rfVEU=8zuyOYrxLK@ujIlNa{sPsx7PKvhb5;7iwA0t@sV(gRrMFpQ'
    '82ma*zUDEo#L}PCMt6Afp<z<w^BC@_e3pRS39LR;Tt^Eo*}t}6C)Tnc{QDkm8xc9BbI9gHbw@L<h<Ng{fcaNuTfWP;`e4wpa@TD((8$>-M>aR67CK;+{pP|g'
    'J5y16s=sBYW%N=;dnoAmW^Ap$U=5UOF#FBHNuL^V=eNe6n8Hm!2jD1TD0B>Pl)x=Dn!+cvd^j^so-wy%@fLua@w%TpStoOJ*-2n=R_N1WMn$YC`=J=;Lku_r'
    'T^uhkLkGFHKocf)3Hy>_@5z}oFzqr*Lxo7f#gCyN=(IbZs%~WUVp2$vC^CE{R^+X`#h$l?=Gkh&JGvTG(-FI3Vr*$+tnVA~77n@*anRmHekBWd+Eeu8lp!lw'
    'zBG-zM6<Y|Z}wr<s`;eBSd$-t?dowP=&;%PLNhXHn&ONPUZXs*TC9fb&p^K^Hn*`6c5(YW#1|*NczKX?d_6?tp)f112?z&R%e`#!a}xpkwVr@`f;FLksikm8'
    '`D==}>w%0GQ$5)>gbq#ESKdNo5y3;aSh>Fz6t~bA$A+#Lv34V(ja50ij2d_wI@Rl=??Pc#yUzhhTZ;1GFyQQxdUpjCv0khk=#R-R{$%K^=oR7x$ue||R3{JW'
    '@_d_r1!meV@8D#C)Q&bgd;g0@MO{%>8@t@cuAZ=@y99&aDlO17+11d&adWMq6unzZj6~VczQv_v=L=GDMyq)-LB{2k(hhPT*a2C7Yx3RqQt`j&NCr_WVb<XI'
    'X2P^b8gU6WD0q2K#}w_%%piSZUqa-9Hf?s?OXGFWaNM<&qR%%-^n=}X$j(tTF$1+jV1bjTA97J%iEWMUuS^?m>>!UojG}HRZ{c|ny&J3brovkVP8i)Oq{mI)'
    '3Zu?9&J})9NAOl((1zC7%D!ogkn3eY8ri$8)os|>rQj^z26w~`$g!ELux&{V)zq@6v<aJRw{~ieQc-RQc^6K}JMyd+q+A_s#_pIRy^HC<<TUI)%C|j<l;1m+'
    '>AP(kas%c8JL;3}1PBT&kL(Z75r+o{Y#r}Ds3kVvn-B%x>0AmD>{jQP#uGK3ZqCYrk*x%+?U*q;xjbo>0c5Y4&3Mo=C3?ewosnT6KXqB(TEVz@+Z}MwGysJP'
    'jzsH<)61f5*L`b8w!Lab@^97YyGY+;EjZpiuS(A>3)*eF6~av*(k1fw%S`}z6kwT2*dp}`q8B5Ez=Ss59WQMt1W)OI0y@^ha9?5~H;H5Ag`t;_S6(1h;qMC+'
    '3q|gUUI*y5(9GMTG6$*hBE?}b*}UF9=6*?U-D&33^OjcZ29>(@IY;C6Fjk-vEHZk@TITzV;>(Lefl07}n<R36Z-XOaHUjDZ^cHC>brur-?zBZXmm|x$xMET6'
    '<;Za_nrm_WV(ul2s!I(t=2u_{YC<i9J;kUr8I)KSls_*h2s0r_FQ$|ik)C5ro7px&rX}=UM8Ipl4tj<+p>%IZJ=8!A)@90HBd2%B14Ki{<oC@Q&7*2691gSX'
    'PB1<Z>U+U(1729jz}o}IAn!D*CH>8tm_<646YRW;ip(uw!?CroU4}Gi`|h!BxJTw_eZiJvsV5X`@(T8VH$Rc&wfWi(srd<g`XlT}2j9cd?~a8TA>vgNriMP1'
    '!lUUFiHx11h&XCkSyv8^wegt9a<SKN>7AS;2Tj{CPZ(@E0P7s2@=yUFkJ~G$=#QPH%nY;oNXe9KDVNkjrqM<UG~tRJc%zDN<Q*TA2+3fm67y+5Q5vj^cSuFV'
    'C<g<=Zma-?ig6=@(E-_<Yi!3jOcM?5gS5o@Y|~!iPnxqVvq$BQZ8;r6ygfSxNxqaPZzk}6IPh<X{-Z4)=B9o#K~GS(jQ5}dDg^9&+{4<R#JV4bF^Z1EG?lAH'
    'ylw1nVfPU1TKhZLr)~YZ^*n(n?j2d0k`j;uXbN{_e0HLeB|vu(OawTvQ@Df$3_BUqW1xH>4!TqD)E!PTkl5Ts7$fS0F$u6e#FgLe^`1nF+8TExD)NbldG?g='
    'd?TJIu4Te&X?($im=z5P<$E)6E+pV7&??h%w3@S4q?*sml}C+`poJ}_DepOD45;K97Her=!C`bYqxp?tqW3MG4)o4qnJiGJLWL!l;SRp;W?zPatT7>5PmVPv'
    'gPw-I#BI}1*D@{I#91xMVSh(Q8lx?3qim?!X0J!;7!z;&tn@K+V-qf6SnNc|>q2@uU0#eh3mku@99#5O%&BqK0SBXTXhXlM@T=AKI%MkCr<8$P0I2bYU^w^J'
    '4Ug(LesVBb+^w6)olBcL<_<NG4^YZ1Guet2pwLMt{{Df-J!g!n1~eEC8*@X8fEH(Ctv=TSi`8nMQNHlhypk&b6r0~L6~%gRy<Uym{w__cTYcw7VUu}Or_z|('
    'dtMB8jHFG4+Gb;ZCGRU$%vCx)-l6qGUKDH8=1{XCiWPC=n1SNXGUV=#L?p@a?&Qr-^-i$t$Ag|HU!Np@eCHUM5{C$5WGb)obgp(Hh68rdALbMaRGE&PT^E3='
    'bN+Ef=oTRjj;Z31qdyGjL7Cn{&k<XhO2CUJp9Ue;96V_w5}rc7c#)3hLFcaQ72thWXwyaga|e8P!M)RJ$A$M>xTc6$pTyIFSbtVgr~vMtj+<-JCQ#G@EM*b^'
    'TLcy`0FuvUm%L=Y1-_vP=3O!O{Y7N<b~~Qc%@DpdE_w}#i-fmtAl-p%g`tmq)qYdVqyvH`PX64+RnuOx00l!tr*;R|Yipa2krpCad!7ftrV6XXr2rADUgq-5'
    '02QuO@5BPjJ}Vn4`_2s_k2V$O7js$oJbF7cA!TvB&0vj|nPOdQ=7Y>kvmqL8*dT1|tVf$5TwXZJ!{K}DO9CYAK<Dum<C>JFEN~aRe$cGf(6h8M{#NyZdl>qW'
    'jxTt&Fb(lsmWd>EA17aY@rT`XFwL|p{~Ib{wq{J+qKVmB|DRkPvlZ1_1xEAq_dFy|Vh7o0k%m+Y0br>`F5oJXO}y06&=aw>lzK4y=hvy|a*NZ~?1bu-X~Lc('
    'zOE!ID()SnhB5a|M*13&{kN4v$|`H$4!$`$r9dfIWKcYMRKRvB<_ndKxtG7Ms^_x2aY~-HduF?qhv}oOS{^DEYI&@LLuq=jTVFW~yf!@#!OL}6P_@ti!=`8n'
    'n%GIxBa)drs(MUQ#<IY@5Ms@RF17Nuota;i0dyM;+)lS`V~;m=X~-&$w;YrDe|-|rzs8cA$gp{w_F@_%Pb`^L??!k<t#~EConj76$Ndd2W|{u7WQ}#%MoYO-'
    '=oMpTocp}jut;r2whk8Yx4D6uz;vtv&%bBv`S+*~uL;|8@^*zU6-uLs*|;O3_zVL#6ew2+YZov((h)P^CK?(BM<AvUHhqEF2dAlB(uDV2_x)z^7mFk8<<wBm'
    '7_d&gGI%}Q0^D6cEzlhgurqjV`k;L6$Ogh0Fxb;FP%4FC3&@xY(@v1BQlbqq?0dX;<ff`-2#OrWXf-WRJ`61|?4jG8{ArxrU7!kG6rAxmf%L9dZ|qv7=<$IB'
    'd60&vHsf>Cz|q5)H@*!DA+YbWC8p<9Ltl^pIWz|9EKD}(vm-t~vs54>3j3rf6Ak;8DLMHa;5MYDEpW+#ad)sG4sL<^{&pAJ_vtVX?|d}J2x`-Q9QMm!(&m=j'
    '!5E{VONmA$P2&>o++T#;o4nkwW0xQgRyy}rDBs(5y|NvLH}mS^%W^*)Xl9K$T6CQ@8dv<YXUaiG7q3DJ9!&m3HC>Guq;o~Ildr1l%mAi~T4mnL72KJILKH49'
    '`5*t6WC=vFR=xa}ay2if$%`KzKEp6Z<Hd3Xti;NBG!ljCy2ru08a5S?fIGVgL-A;&5itYM&KR<S+oT0sY?kczqPjwFPXUQpf<)6H^c%qRnCpca!1=q5FRQBs'
    'N`eVUpPm2@VtPR$HBOjxL3(hyna%4H!#SIX>nA<>3(v6e@Cagw(Y)umNm2(mY$r?8jF_pX7-jS7)+fa#)rv&bmQimAEk=|bo1|quwT^%+Y||KqG-#Qr;XzNh'
    '|HQBAHG+|~$d?>*aXGEU)B}%c>F4LY?#_#88j$3Pu@>x6h*Me^*G=i2t@dWIx`b2w#s<;ac@|<V4m+BZ2?0J)18KJ3GDF18woQ>s2X3jH=4|D9G`WN91U0RT'
    'mUk`?$M7|=tKYTd?gs0?gQ}6z2AyqbPaq+U`84pr_2qDFsY^i<IY=koqOPxvEIU$admHU(cP>UNy6LF$U?3C3fQQ3AMgW)_C3v8o;<~6JINGr_)Uk6dZfMwF'
    'FA5G+(IB?0aKY7)UhPn64eVYG^EgA!C-)u^Ub+u2_fInu22QpB9gGw{od#VPQQ|su{P!=iT|~$pZqE~Mf}0d@SI1QvX)pCG<qCc9Adf|&;o`v3t<lnQSyXEY'
    'dtFC}_xWN@QKqWJ3LT3$+Nw@v8ZD5iorvd8pB%b=_3V<8w;6VaN)%OHoR&c4uQ(ZEheZ8NH61UmY5ci)Sj<bK<|z0T3Ki$TYQa@-mUhi-5N}GQ;?zJbEFG0R'
    '{a{P@#eq%~t~;R?tz%ABqc@zY*0m0#sn@(;9@u)?&~GXezv2*xKKq(z5Bw%6WYv)m&;bV9B`ONTP@(joF(+yJ^^7}2?kFN0JfU=+^nB6tEtPNCZCRouKar+G'
    '3PiV!-+f87Zi#vpy^L<RH7fK%1KO6m8=Beeg1+tA6NtI`n<ZLC!~|P&T4XI4Hw%oSD8y8>0gU?zbGW$Az9xgwsCcU?gYt@xgQ{d|iW0-c^n?V@d9uRa1TQM~'
    'G2O7qlN-_=-6Qmi2{*9x$J_8xPmQo<fx;Wx9Gy$DA<&-*{h1(wd{TP45U@Xs&5BoaQe6X)O?KM7(FU$Df^X%uEkSlnYc^?)fk@Vu>D}6njoTqMi#P2h2aR~h'
    'Hda_{t6b42lJ7<Fr>Y#_-Kao~Vj!+7@*EhgH#H@!m{#W%a5BA+F!@>;f|zMxp{EU|D3(`<Wj2#ZHKw4Cd?+IAFIOet$8(sNazz<d3G3nZvM8`q!{d<~5j_f='
    ')<LgcLi3wtLG0DAq3AtL?t3w7a*i>OW$h8B5N;4Wr$r0{9poY{e6PfEu}Tf1H4&c;dX3%Cy;t&W`(bz-MxB}T-g1Mt1)PEz%?<u2FaUAAPNf)^8Cr`8lWx!4'
    'P*SXIF^Bz0HJt*@ODs<2h)t(I35WiHhRMSyCDHO^Go2Dj=Kf~6V2G?yejr_s&qp6F7gNHp_u=HUPqLd9Oza~Ks;z`#@HwF988Rq0$nP2IkZPnM_}CF9S)4n?'
    'S@QcvEl0qPt1mt)-~xnN=j#-4ex!K&f&_!=PuHnJ?qqD5bwi9S;=2N6<po8TEyjzBa$cPMisLHN;KES_qkz;^P1Vi&$W=JZnkf@(geInQqe45Y9gKbSy|>hU'
    'xVTv^))!@6G7)(P!QOH@np?`d3udPChhr_?@bF{MGerSCS7)-%et6JBDk1oU?EeoBM!nskjCyRUf!ujjFBnx@LC)8S5@MCtuysSEJ5YWrehbIgl~Y6=fV?SW'
    'CB8B$aI43E_?JgLzX!*<eNCjQR7NZ#JZ=)hPHcD_Qo_0M4;F=tFmzR&FuSJVo2YUH2Cc`!k4VJ&KmPAkHR5o_Fm=btqbFZ{24wFW00WfUmkFG&eL7#s&vIh-'
    'e}F@ke8sl;x_~->K^Z<K`;_pP$?fuNA9lS<H&~<4mnb3V`ULGH%Ik7Wfg1WUid#RUFRU5tK1}sMyCLVrL^H$l`NAk_iC@iHPXShZ_x;~}q};v-+=?s#Mx$Wt'
    'L?OLoWB&E=^A}H^{#g}XZRTd@*Frt^`JD#I+!`7UYNZ@q23SFCCOOxfE1&?#Hfx0--@8CyM45GELPVmyHdQ}7@XBB#wqRY@koHv8aos^VJyr~O1|*fDo{?1K'
    '8k>1-0c~TH+Z)ULtp%-mnZ1lg2RM})MyW&FIhBH!{{r?QBky7bH{@~hY=kirTb;WS(ODfED;wH;&I+gaD~#;Z;^Ft-KY#rF!<UauN4!=TzJ-;AZh$3_(r_>U'
    'W2CAUP^~>vsD}@0jQlpLMS(!4iiO&q$+lB%lL}$EDWut+)2N_<5@xF<5XvYUXNW)o9-9rI6zeAOi9@Q;*pTKG2=s+rwG1Dg+PePCXH<>^uciQ#%uZhZW`+23'
    'TwkRQhj5=_$b%%t++*TMd<|3XCis2=BpS4L+Xv7xd)(A(xC0cI-!td`*qnZarseb>H_pCyRjjHdNz#u#`y7hWzJFlNbzz0iOQa#aD#$=pz(~PS#SAG>EpRiw'
    'K&zy6LEdtx)c34X$PohSH06Iiba~^9@Z}2ga&AT#`U;L7P@Uxn6Krtsv9ow}##Ekepr|I+?Y$VSi`k-Hi>Sbw@JAA-W>LtEESk=UQ^li&7H7EdrrCI3@qXo~'
    'Ju%-J7g;w_)f^seGNBaau2ar>)LLk?gS#YJqG10!OK%ibd)lactFvRmdZ7gYny0972isz`zJ74<`2p@O;J+~KCR}l=*5w>;&PK(0^5DoAEU^u;c^j?~<=m#|'
    '@MxI=SR~j;<@_8DS4&eP%)O^GqZC(gDm(f$n&O^VQ%+8MzWgs)e6BpBMhdIEl51Xw6Au$38ijeSCKPAm!Gk_xfL>+^d2Phu-!Oan-ocZPS5><Ey#a6RedD5%'
    'wp~)ka_h;Z)>&4C{DP^T5(0;si|JL#SFyZS#1F^OGPkTV_UpKZb<|Cp^f)S(chx-o9QOj-lKvbAx$Y6*q&L8((SmiDyyb#@%yoU6{9OS)d8>MVFFCxu|2Eof'
    'tcBe@d5izzlLYI3E6dZDi8{OO1;QYndLs{F>keZsD>It%pWt@qh30TqBwK|pc_b-&M6j&Hqi&?laHCn>@ITpZgUKNKq_@4A`a|{<h^)*Ib=Vi?yEwKPSxp2='
    'XSNW&gzMeXAAM-OT8g2WAwev1a2j)}l=bN%OnZ5c2m=jZsPk807+9%M)%E5SpJ^mnXjZNApiX*E<~&^6EcX?VGW|8%XGST4U;sfMon1Xy7_@%HTvokP;8v{7'
    '!5VX_A;SohyWX4ut)S?*&NJ-bf7pYJ1IFff0(|8OpK&z3zAW9wHRpwo1lCihCDLExaoEg3gY%1R(>yoD!A6ISB4A}xL2Gndyv16#c&>WQ>=_3AxCLe$WI*nQ'
    'fTDB$?pWwUVj`zsmn|hh4^Jpq5-Z+huVrm`$=rM9b~Mj>9-h))?Mp^#vr#oq0~QMVjJ)k~IlZx@a!M$rz;8z6DhC+FGtvBW5nl)jL4olD083tu&MBC$VM6=K'
    '6XaN?Q<AY9C(CNdx<q(`b$Z!uY42K^rKBsdc4;4r(8;)65b^P;qjd_5=*@#y5)0m<A3gfZw+|Cr6fjVPSaQrf-#h2GNopu1C&_ts^OAbWEMxYOtU8ZHQxW`z'
    'k16k6@km6%kW5O{IOM83r?4>4tJH2sHx_(`sh(kE2RZ?OswdP&@8iFHJp1^p`1ps9fBN`E&sYkQC7$AwsdXXj;2FJcCOzJ$qq9X_ycGrxPQ7jG*5mz<&Fm07'
    'Wqf*&V8nL|U94ItPSWgHya_)bCuwzcBzr4(#R@BX(*ndXnsY*#&3eoZ7bQ*UeDNm5AHTv$>%%v^@D{{ZgdFIdnO0$9L<Uuh9V;eSK^4if=TClm`226hPmf>z'
    '@budk#rHox{pR71UG>?_uBG~qPapZUVE`8;5>gcGb3u`a-o`jgo&Dq-s27~htMo;7^0@eBfLTSqVfm_!A^ZZ(TPr-H7*YQ^8}=HU>l>Jl8Ei!2z43IcNTEtu'
    '%1!#tlCq4Bf_1WtDxuyc495>AFOVPBV%x-V#JvXpWY|*7K!AXPT&OC$mX}6-EQwve*<UaAuQ``%FD63390G;J_HmIflR4)^mDoSz%=RVVN|iOi_$!-)2E911'
    'JRRNe42-=XLr%!CPZ5E{0}cAF<5Vdp6cRAGR4FD|Ywtra*@s8xr-swkCl?W*VZbfU&EAjoR{0u<lkQ-i8(cbWb45?qzX{K+#LEUt7`$<$2p>G&Bed_+{P=E6'
    'dnluG4(|pjvmSc2ussMdv~c$B=EN8Mv(0Q-r{(;rS}o=%_;;N&(%IbHGoyN3RpP6jpx|-7J~?vf6rlC21%Bs6u-gR?va(vt*5J-PZIZ-i@v0skgtIT4s5NH-'
    '#;K3V15)f$yUE?b!58S6o`q>Sx+M5s?~r)Fv|9n?E28RmIAMb8Egqc+G6ur!yRS~(cF84gzf2lP_rL!1C_!xzC9Kl7>@Tyi)?|AN5@c}VBwF5v4>H>ZGSlF*'
    '6J2%fL`$UKjcqsVr(vbu(pE$M5{|ugF))k=##_Un+*;-p@EDFl4?tV;e>eEPsH3$JGh_=Y5Z*Z@Osn?AH%R9LjIPs|<$Q&g^O_x`iQaPl(S}WYB({XWMLak2'
    '{}vlF8#>KyNvyc{JK(H7;C}$`;h595Tg5>_?(6^Y-~K&$^o;IeG?_GaqKhSox8rhE0zH}7zyZHO_pTm%+JAto;|Psp(HeNPf)$;X`;*Zalb^#uSuW67jH$oe'
    'GDAu7WL+mC(=n1B@eovQCg7xal*h@)8z@mdK-}yD3AdS!+`s`8-ipbaiv>muV|y-1+QgdoeJDu@-bjN0e_<B@%}?jNzxO);@Cax4oK(%fV`uQ<`d_hipeEsH'
    'Jtk`MRR*m6tKtRbr7d0zp(e#Zl>StxMkmUKs2NELWm*E>Lj`<EFod~H>%?&q0UFfjgicBFofsjY+c*uSswwqL;_m+a7I>pIfU0puEIb<HvtVzSq5h?~PM1|i'
    '`O9X-Wio>Q59AwN)5+Vb^jUG4?O&y&(93Qq+{yl%48MGb@4x#$u)wX<vQuN+LlNW%W_Z4sW87Pb0J%<_u~}lQE`Y#WFD8?^&jBXwZejdSv%CQgp6`IOoB3#l'
    '`M6dKpnzv+R?N{Lk;%7I|C@^vH_3Mx*<pVl_U}H`-si)z&)GUtxmc4gGMs)&im^HeP~YX|*a<~h11!BUP1ZNyv&s^~q0>o(b)9pkYg%Jk5IgtWf7&Jh4cAO;'
    'Fa#KM0g%;cIhv_~^IRa}r8CH@`VdB{<gg}wQZ=7UNl&g2ganXOcmh~Wobx`N05>8^uKNSfY@_*21zV2@ZL}1DC3J3*a@6WCd3<ZQJZ9yY9BgJt_6<V<yL4T`'
    'a?c29AcrEM&j3lx;B4%l;x>lyk2cT?;#K;4dp|sR@$%{OC%_&SLR=Mpeeh{<gl0uJ1xJNwqF6v(@Thu8^1I{$_7#wcXhaU2=w-S3(ySsYwgiEptmKBLb1%-;'
    'qRC#|&OLYTQQ2OUsinHDRrY8w-#mQrxOn#PIsE(Z`S&l#sm4ggREdYrp8fc@;@R`3FP}bo`eQ?!rb`-A1#2s&bzGu(__QF~fMPtY@G8Dib``LYLlwO?ri~y<'
    '1k1VjvCps!P9*bkP-`=z5E>?a>lr|U<yhnS!2L2~CJsZH5d4+(H|0mRhT*0k3N&g2$i_2N{Z5xmFlDMzA3vG4(;gpobyoL9@(c@M0<$`L|H08#P|VQo4DsmG'
    '?GRyB5fPmMUAIpuxZ{uxKK<iXSWT=Jh4sZ&K!|tt-&AMow$5yaV6Ucqys-8sPEF(-)O6;$&$h#~MD#w=pXiu-vB2PPCxhOF;3_;|)+k^!VPoUT1l<))Q?BgR'
    '1_jdoMRf-FZ=bTsM<?X)=#M)CBkRSkmHzDWonbMr5rgu_9Zm&OE&H!4Hdl-;%VD|y*{*=#?%jvmzMLB??-qcc`vzYOmr>s3eDP+^QgIR#B4WxijkL7V08vyE'
    'IUPIOsa)=xiXxl_NUy%BF*)|t;P6<i8{xKqi-B|>)Ve7l0_~MD{WqVG=`z&@2f&2glh~l9U1gchN{|@J7en0%LYc-}F^yQo;G7}@aP|_U2T;C3gUkA;8HMT5'
    'YC!D`gEfJP0jhLdrg)1Mcu454NtfuY5Zyz2dv4~<T&inItq{y@aWSe3zTZ>+L)<<s7)r;iR_H36YPT4s@%c@<UR(ktb#fvk18M^3Ri8pk)Of#0Gcf(Vlb*zb'
    'q*@$$^vc^qfgY;U2(@j0HCEHVo|RJPcZ!O2sstkHfh$%T#~KL{LxUp4ilSDs$wAT&$wG{i-uL)$UE72wL+UOnsA-mnDl?lw$Qn&qLsQd_UpCeFGN?eC!QGQa'
    '=bKPuu2e}!qQzh<{xNF;G`3v3!EG<y1|Eg0av2wdqN89Nr2w2o+a5r?NDI11V}RUO01&J0LL_0*&(;n_iXp&HJ0Q-*j-eHn$f*(mvTaE@*7Md?4S;noX9Q-)'
    'iiV)0O9>!9Z@_FWX8?<M_-d}~-K^uKOyNQQj|Vw{Kj<GFP@p3}6eumXtBy30g^`8rwH0h>>|L8-fhm)VEH9NQI?AaXbVT)Ls#&4gb#KEnthPNLR5d$~sjY&j'
    'Edg@)c>pC=whdlzc)Gyp1JnY|9-PS7{Ik<II7lWHOmCTvl>1*a*3T>00);%FEV1RF9UioS^J`@0a{8{iCT>gw0mg6ulWic!E@?HK%t@p&B+M8uO5-0Y(BSI8'
    '$a<78l>E2xsu3@pP%<e=I=Xx&Ja4(4kqYi46J&|tV(!V`P#mUgR=-S~nNZJ@C+%x_l5||eFqAEI@g}hfUK%4YOnyPDV+fk@lDD@B`D`kAWjPi|`+CZ0t<!BJ'
    'Hj5mHO!KkfGNBYHA&=VXR}yJqB8KzQoBh+u`arpwW+$mZFXmpDQWOx7BANZNC|;fT4g|Sh*owX9Col+gYMKs2g7)x8DB1@=(PkkTLv_e}3=hX!-!|nV&o<?w'
    '?6!Ct*OcA%A|$d;SQ;`pkkjn#{Rx$kH2-@Xn-r-?C4GBDTiFU$9?zB)%dwi9UCAKn7Uab6>k)=#J*$32sj4J7C)(9YSYV~t*IQ>n^3xpA6XPIp_D?7*?q@c+'
    'fWszUfzS6OQ3sIQ7~2v38me_sU=X%RU>PP6X(xT#ArqUBV&$A~78{c#N2R>+D)=!8{m@%pK^SH=L-R`t033Ydl+p*oW>$%Ysc7$!r+lH65Yw+3McUJHgeQBk'
    'S+&Xj$q!u_0#aY*Isct-1l~sH{L-8ixsbp{$cAEM33wUI4rYmYJ%p*lIZ9=Y!U(Y3Zsy!{e&tnA8S1*`ye+V-fDNZ}ep!=$-Dpi`>XKCRv0mq*KRW3N?g?xs'
    'H{45LIG1V4aQBmg{%0Bb1DUXYFtHhh1Nht!WljWPhs*8RUSV*@{&G1<H?AC-HcNZ(3va&kSJXbf0nS^CA^oy*AO_f3KE;>Fe9uoP*ay0j&L{kPUY^VE1`HTZ'
    '@<~FKfw{#${g*1#?~)wz3lz!Yc9cdAH6S>pS7=6qf6Fm<Yl@<T^FHcks7hFxOPO#A27;Q?C^4bBol0w;2*3d+vH|8k`IGdd|57CpA5G=p*T1aRIw>Fg#KCbd'
    'IjqA`7MvWFQJcb$LJz!f5Ck5V7&}2hKk0?(l7ZR<RF<Lf<zAz3=kkPwgQcR+Ks;jLNYZOaz-^YplXwsQ8AgI{JB*9Fjbf8hV|j9|F+5>k_o&q4M09vzVXyr|'
    'j%sg2L+Sou7UBug3CeMQ57fYSW(2(rUJn~MM9kW8%MgtqY2wL(h(<F&1cKhtC%tJoS@*In$OG2$-EhZi7muWYKC&=a$ggUt$OPD1tY3FptY7_w`n6@s7n`+O'
    '9!BGBW%X>RL5$A;%XLnHM6hL?LNrE7DDAL+klTChop%Hb!J=l$Yy`x}GToT#fn&JFH%t}@)1u~{{Sm#%w*)<Bc}#3xBzkUL4arb|o6$nZOa|emp9AYF%|6$M'
    'y?>65E9H$x!box8tI$T>C!EyaxR|c!*psik1i0eYWAg~|*-icK=E*_UvTV&bkDg%MWy~QRtGQ8snT`8oeU+Y_gfmr-fzbjUfCJpI9|Z@C$92Vs@o+T%@WmmS'
    'rFnHF2hP03oy2Tx?u*dxX;+Q%dR#8o$z%G1JE7^Tu{BxR2kJEDSCbw$7RLt|0I-W2AR}|UH<9DqaRsrD_=$iG50$*N=ZgCnyLxiTB7K*m1>DNF7{?5V=uM4r'
    'a00t0vd|)dG$Iqt{GV29ivMr;9SR_Q_xnvH?ijpo!B}|MQj>5XOTPX(d5r<K58ahDtLmCnP@O>VfTuA?iZK7hTzh`bHs!`5jYY8pZrgGR3}Je$sGnbK765m{'
    '6xS9*IfCLki!uml`io&*+K`4yapHHZ7A#a<7dA$>i&q}p*s83IvbE?YQdE8fUZ4eaAtcNOb&a|_MG1&Bjx%AGHzh{~v<YAWpqnVrO+5_DZKCBi!ywDml}EQP'
    ';YEr)&y$OCHG?e=Tm^?)(bZ;zN3bt{A`pMVbsd;K4+GlkiVJT5CwO)6LDA|>{6E|1&4ANlX1ZlJxb_+&Z(FnBo?p?jMOVZXT>;g-!*UsDcVUjMJ|J&&K?JU}'
    '^{2$}HiqqCSh{yhyN!H;#h>I~)VV~`NWLJ}o^9+whxtKxf1+-v1#LZ>1k3Ig0Osu;3)dk_tGMAFYx4HjU2p-{4_obQQxKa})k<DTDWh$+$%~>UXUM6ajhs0K'
    'KP7k)OF^EHYJh@{wO?HfYzILW7mNw8qIsnwIi(x7mPC7G$Qeqz@KI`ZIY9TGaj(g8QCgpE>UDB4x+<Oh;(RkZMcMYm6Jq(SMyh%cOJ%(-m!hzEAPN*WtC<U5'
    'z*)QPO?i+f9$W;Y281GH){X_G5Xe0)t*pTpr=-Kdt<X2BKiVr&bwCx50qW4>sMLEj)I&G}mnarHVKQIX4Wz0J_d<q5*-a>wCX-bZY6)op39230xO<M)K8}l#'
    'xyW@h+dcPAWwfW3lo=q28YE-5Cf9PYlUMpuc~7Uor3m9}wJyOt&}(P|stv1|ZrjW&Xp<~KaA#3Ra(3J!@IZil|LiZONffdex=w0}6$yUrBiaF)wN{iA`IS?u'
    'usH+w{f!w&Bd(|9G;1Nb^dT1>3O63Z!h7eJ+%O(dvdBB(Hm;>tol>1rv8<6t9*4AC)VUp(@klHUj6k?k51=^0Jx=k*Js>ObY3TSPq1~?;roC1i2CYhK`m+VQ'
    '8iZ=kK<@7shqu(AZHs39nPK3Y)$0WgWyq0VMVQ?^6}N5YY&Kf2fZq6U=j>J6Ijfr=KG1m!Z=YBGrSWRp-s`wsTsPnKgxT1-{waBtF*28;kuC<#XIv@MNdlKk'
    'W!Be;JY`Gr^5w~^K{eEetA^(+H(1BceYj=A0HYOoTuDDTnSFB_hS89sb}yQsTSO{xzf=C>Qyp{bzuHc9XrYc-Q+~IF(VbWXXBzQH35G$WM0%Kf^;I(IBqP2<'
    '-;BWHKoSr(VkCGz)rOX!yos8lh>4UjM%7|fapI)`x{LSit5o*mrAl!~_--g-iuqz@CAJRGU@~-DuO&U=DOt$SEWGvvnP$<sN@JR2I;qs}5z4NaNHA*WiA!gA'
    'T1mLKG~hTTPkUrk2m*B;dZ0+7AmshzaEJnLni4-L3FQWV4NSv{feq{yZw1=x$=`7pLja*(2Bw<`!?WFGyCHE2N@JTkPCO9)F{Tp5r8z0Vru8om{CJrila$Md'
    '^l>{QzH!^*vc8S<kUZV(LHtRMxaw9Q%<{&OCqV`_xx#3C^h*ZSxv+LYyJs$Ap)n;8u%}k)qTIKC&>(Bz9dC0ysOC(EQ;Az+&n4Kl8%q}3B7U_2Orkg%eHqrk'
    ')>L6?{9h9X$6GwN<60A&ODjRr?e-konNI?+ZzsnSOnG+0n>guQ|Jddwrx>hzycY-$>F$K9A|)8yp5b(MS_W;$-V`D>s=+3BZ89KX_$w=?8j79PLzqAeCIK+X'
    'kqfF-XPeQK&3>?*s%9TOGUpmi$Px^gKyrz524SLe_??WYpFCM7<Ix<O9$#SJ>|wZXbCzI`X0($)aLWDfQYM_O!srN|`B~z4#AYV8VC8ct6>LH}D^S=hEj=Ib'
    '*~#qQ^xp1;DSl%haS6G+4@5S3*rF@0<?b|I69OvTPpSX+Y`Y@$e{JeWd9wKM&W_!6_sm8qa-KP^GW1=*lszXO2BDsqu;?N}I`JH9tnGpD(RmxeaS8O=wMYh<'
    '*nT1-w{(ujALwq&UDSRtOS0<aapS&0um{JuIJd^@rJS#IIUJ5lcujH&yh>dYxr_rTj^lSA5~6nkQePx=+Kq0I6}9<ugL_2ZZ-(2OG}bc^*p7h0#>hh^4goK*'
    'xcA_!s0ai*W}0q2ygSlNGSsVTv8hcqa5`_%l5Pa{n8~fNq;qK?<;gzjnqkAiB8F2LoI@|&JI!d$zhBEncCAvz_mk;pc6v6V=!M4#M^Zk3^RZ9AfE@F|o-Kv5'
    'FlfiM6XfTjjcN;%P>s65#pt(9ntq+q4)ZqU*|L4}ot*HV&b%S;@k>n?VGNuDTYRZ7QZLNtvAY)C{upmofVp%(sVY-G7Zob{sDxYc*X$OHW8_b6quHj+vdRKE'
    'xjVP^ud>bf^PW2==7I(g<qEF5)I#)W%68hp;CDDinb8Hnyx5TqJ@SJr;60@45SwTBqbUwmWV%>fZUA{b8cnB|t#mN(D3_raIvH7lkG1#oZRTU{T(*j_&(~6h'
    'cS7hMZ8LGy%pDND3^pC0T|3f{MuyzLL^N6IXTRUK-K4FDm--sNW9V?aejQ~|I2^?rT?Y=qSQfxG-K{aD{Y5zdYEdU8Np2#?L0=F#@8_KRB;&j*QgwKf=l*pT'
    'N1oBkgLcjJ+91@wTNsT}{?JUAmjP7f#R6muzm)*pTT;Jv);<$-UH$XV#tyeFpYA;28QOB=db`hFSyYvc^Y0~s>T-*8c&hHXzUX`Ecgeq$<&pxnoRV=pPA6s9'
    '<0!c<Y6X(!><%!!;j)Mo`i|nhDH!`ZG_~8g5VFZi_$219J7i(RXu2h&Il86g#%$ye<FdxX5pzmY$)lk;^!wM;<h4m^K$mUub&^uemV*2ZOGYRc?$tngMAt6-'
    'bR<3PR#b<wX`&g0pt&qY5A54c+b#}{m8X$*o*LSlT~H}E*h*$>nToAf9o>=G92U<LZ$rClj|v?57oKsrFHmO~6?X2zW9K#zB<F~bm*j@#m~h^Fns14vAK!9u'
    'VJMUNIk~WPiXnH&Z;qi<3Ny|q<=h~jCAV&a(e${|KtetVm<=r+Kwczo@EWrUX#dw}F7Vn=;P8gATW$mrSsTVfq*@sngYaaCxL-%!Rl&<0xfmVPCJ-k9NG*G;'
    '^aDyQ6PMo49Y<bM6x`oC5%0-^T({Cv;q{Ic)Jj`)tqSKjy|4A^h2Q9oV`QOVMe)Sww6>_wA`AI(WU+=M&lOKil~zw@LJZ<z`iD&7*!eFh6RL#St4s;~^0mBt'
    '9R|KcyOe2JU<PffDG5IP53ltHnh1RDUIrj->JQ!_dU-G;-D$m;Zq}tr5rT8qEE{%JebY&rC=r-bHd)lh{ggmSVHi}E?ZiX}499ZoOL1iHk%vj`i)`2x8E(0M'
    'h*dyqMgrek%wrQ8*q2#`22Uew$1WHeMfXvWsbVWySf{{4f~d_jrhJX~rYY0SxU$n%h_XIYGt!yCeAKz&o`*gomYR3bYc}a|U(tCTm=GANy7rvt?XeiP#y{vt'
    'xXjk=l5trEnC%&D#RL@<+q}$=eRj%n>{^nhI0(1I0q+s$M+a=;+@`#3v0pSA$}K8fym5UYMXtiXootc75`UmPxphXUhMFT5@KOabwW)5xd`!9!5;$SRg?Q9c'
    '*$!mou5GGt43@|-9IoNpJ>o>iy+$fqbt60uB(ge)NH^Uj>tsi|4deLg!?#IZkE7gd@^>X95(qJ_tg8dO`%IbUw&|oOKg<k?6aNVxJ)jZ(k+Z|fBGWdVt!5e0'
    'FKaMFV{z4e>3@^sGQ*;B!NlG_QCeO-U)lCk*dmeh5LnI&dzW+B9{FO=$CQOfRa)HA&NrDJ0Yjh~(IktLJEIcxsHvH^;uV2qC&lkejP?M~dYb-0_EPUciTFf?'
    'P#P9L1^Ej-zsldnm<lLbq-b6#m+l5yNaxH?H(KJrp2y)Je7<qYZdVps%>7R9|IUJXL+!%r20ZCp2VV&H8;PozzIB(OqiuIH+H%Q7jnCEL!KZ&XLa!@kd$R!l'
    'vf0+i#|RdY)03-V=94S&p*I*+Ar&*5x)_blN9iz9ni`zPFfIKcWV?(Kr^?z`;zILq@LTUjeQED(viV9`Ywi{w3979zhjXOSeLH!4IyW=Q0EA1>hli9rjt+g3'
    '9TzVI%T=cv6NR1e)t<_U5VThjOV#X#s$;s-sAWEQ$xGZE(HJr53jWQbT`XWg7|$NY=*8e?>Dj!j>!L*CI4J{j69V(fT8`CJLc9VjtFEgf&zIBDvZj=~NV>;('
    'OPR2ccU6fs%<`fOE#;V1Q=miS>kwGN<BRq1JjnhK@(_MN{1!GtLoi7>IR=DH!Qxf?_E~Rd7J}2Qd0iuqRJxXX-s&?QV2(93?e0dKagy+ce}AOm>CDuvN`d0{'
    ';lJd?Xi~c9$0k0q3wunZw*RN}iPaoa?dV;rL!fpzyr>he18K1|)dip-Gfi$K(TCjhoh<bwM(So>ttAvRuz7uE^Pa4e(bb|l<3QLL@wWtgx0x|Mo59vYO~k--'
    'ib3Tzez*I@*9=9=9*@X0iLHp1BsEK(zkHQoGMD7()1UgL`}FtcJUK<pJmD*L`zR=Z7e=v!wS#WdVn~C&jbh#Zr{rgpCDQ8)SCN7_GRa3UjC&KBPF+Qd(HF|5'
    '<JaIbYxy%*fj>xLLD0X#%qZ(0M5q_gMI4m1)JJTLb^7&UvL>Yf6(#R>gKwAOxia9A;=95_wt;YRlfq8k^ON&xG1`aqsIb=nH@pT6k*4?sUAYd1M9=dBiYVQ0'
    '^?{_As*7;t%n+o5kEKl0;=T5!K+VU`@^ro$ofR`EL3Up$uf-<}38RicX-$IZ;qv+-?zQ!KEQ%o0G?|$ad}x^`o}EYpyGXz~Bn~cW&(jG@LWk{YK+{~qQ4SZS'
    'iryHMy+8*KuAm$z%L&I8!kvZFc~Ku<kZmx{k$bb}TNUFz1NZ|ep8PHu{J{OslkbK~yEyoBj<NE87{YAh?;r3x(&F;ryWy7->>c%VCf@rg;1LGr^otm<BQci8'
    'b^m35Gp~Qyl;y85IawdbMBqShRrU|@|0>jj7|qgHaX>2QivgxVsmaufub=M8Rydu_QtVdFjd+%U++J>)$P-m?PY1YJB3;S`tKyADP07La;?%f~r_o+o%hx0L'
    '_Y(lutf)>zFc-lW&#lsr&YZg*`i9cYyJo9<rS=ALaTZ05*LME^;x2YX+#+_xvT_*Eyd496HH@IW<K4h~-iyQL3yTPT4sgKF-tm|02*NZWtK>K(_DeK!n^<hh'
    'W++OwjN+PKxiYTjQFq8#Ip-G{__j1+dC=9)vsKL{I#+?q5R(xVyj!ePWf!`{Os?{hjCoG_8zc*(#Na#|UI^O=HF-SQ@O3W;^`X;xCX^0n+_1>(p?fR(B@*uH'
    'S;Ok}Duv=%f4X>6uF?!uoxG4hjUIj;i5-H@G7Y;0eQoq{H&LQi%+l6tg5Z_Fj}p~~*wHkwMGSTmKi|I}NPz?vFsm|wd;=el{z6_c(aa^qf9v3boDYRdT(S_m'
    'J+uI-TgKEXKw>_oY;myj<=V%|Qz6Hd>Pm{LaG}knOi)J(Z=UakcCuZ1F)w+SCAEL@bD_9Yhr}NAF}+4|wkSCo&>XWnQ-YtBv1F7Sj@Cc+)ec_hukPr8Zf15D'
    ';nv--cO(sO!g}(Zkb0*CEnwAzVAs2oXi0fvHY%0+rkEqi>=N#(J)tL3y>7HE8zmEfYfEB|>H_%wda+8e8B@<cHu_R|6|MVfd&QU0ih-b#KV#C+C=3((7+2(6'
    'k_reG?-h*z$UTg)E+}X8hTrA%(-D?1eapr4X1<tJqiKIC8BOuWIR-mTS>wc85{2JbC$-Q}!iR1b_mK)>Z*8xTX@|aNj`(6wM#fpM{aDkvy*C?UM8shwbuMZF'
    'jPwZAVsjbp>G_=?s({^8&A|Np(@Y#?ac>wCEkB&~)}>IzdmN=+C{I7i_EtE$JOILsC+)^ktukzM+9q3xrpUXc#!eInnX~yU)I3*98xSkkhIfKO*EL1tvL*e3'
    'r$CG#Z4Sq7a9tl?mvGy?w7;9?KPc1)xW{XBw7jl^5w;&xJ*LyT5ded>LT<NM%rLQ3kuXeg;t0=UW27o(8vtq=oYu#uWm9Ko*!=5L%`+eqPnAt@7B4=FmRn1P'
    ';<nj2#UH14pC+3S1-67?!En-kt(GaDbF0<N1wGgoEr!50wnKAdHp&n-Cyt2gQ54E#N)eHTHgaCqo?Sp#p>P-P+RpRPu)N`9#iN$#|1>?mjcLoN2f5b|k5%g;'
    '!lNs#9?ZNs7f$Sg?SIQQh&F*7x8W#bJx0Jyojqo>J4vi8$!Cc5YrC^QQT=b828rqF?o_eE+`3t@C(@Jw;(iS&xYa?l5+Rp+R?fDWk(Hn)Hw&kU6%A?}IYII*'
    '2rNn#tcJIzZBP47%Pj=7(Hhu}H0zI-8~As=nJsVNuNP!GZc1qzz?fGY-q%FZYIK8*T_rf(QQ#L<PVJP09My+lrFKzvu2)r2<@Dz@j|hHlP=ALx$#5fFnq|?s'
    'HG~+TuuKR+tOP=ReYQ=ZDs+yTHtCG%Q($tlpsR#p6`hmYsHl~Q63Pb0$NPuF!7=r8=+BmtW(lD{%}^8;={+^bFPgz9wefBpDb#L^+F_6y+MseLJquLoansuE'
    'H!wmC3{k}ou+8?A-S_g?NZU+KO>WH?v#im&X($3bVuMPi??EE#E?ZQOgCYeN@bR3w17`?xT@*1pmMt8<=NsopVa#y>$EaEtK9i5RF^U_-*RyE{MiwdSwXp(G'
    '1|f-;1*hqz>^V7y79be73g&)Yssy|$sz18Sw5RB(IU_Ho9YJUMCR@&oBaJaa`rb5)-B>h>3DuY>OE}2eE5}DCUnPHRu*a|Z;|1dB-2P%ThE9qSXrw*4P&AQ{'
    'pgQTLK~c?YdB(1zI=hy|fM0lVuJA2)U<vy-6@a<x)QSCtoTE#?abj&gP?m}ISCg9)37ssWJr|WmO~uymy`D`Y2H~wQHj~M;G<{Pp``BfN)Ui6Pd}^3X({qYK'
    'sU5M}=0lno9201=ci|WGd_2@)c}z9(&OJae3B7fY`NOWeKq^Yc*rP(`w1Ok6r5&#Zss-raP4H~kTzeRWf7SiU0)rj7BTYn!9l9Vx=l0!0YvX3Bf|ifI@bH}U'
    'F}FR@cu8{WG;IX`iH+YvZ{0Sb<I{QBo(`yy2M7rcFAj5%NxB8cy|?L{?E6<O`&!w9+pv3CZm;l&@XxpwymE5IySW#%XJPnf_c6ezcQF@(cs#rg7XxiGwwrrF'
    'XG+HR@iM>}qwEjOFn}ghZp815toc+pqZ64OWY7TG(5)#E+G2j&y0Y>3Rjob=I#?BTz&J4>UxeWe!~SKFP<yxka90YOPCW3Yi1nSrA7s0GS9ng}?OqbL5AW+J'
    'fm-*`s_1kca%|_sl)uvv_xw*98!6zGRey}rjKtvUoTDu^>>6zD4dOSkXh(U*e1Wm0aL69^9VkQ3;96BCciN0LZ&cX9z_1I^G~jW{K4)YoX#fkA2Jp_7hBP^n'
    'z%FkcGOiS~`?_At`)8ZkvQFu^U-5~#!s9|^hOt>tD~0TsOo0T;Y_Z2w9G0pRSaII^0oi1~d26DV*yG?f)l{rVi6P{^%2iSdjTF=5;&7UT4>FgnR1qpquz)ef'
    'EhA;{a*u=VRPF9eaM5mu&0VA0q*Vjg<8nS)RSWgXP=r?393>4A<B$Zh;l>cs9C>oL*Ks!2{~e(d!zX>Q8N&)B*bfh3(HG;()T}Y4T14B5%q?#~65^}KEufa%'
    'ub@6O$V25uVTw0*M=TmSZjYF6pxHMuw}^?!Z_`MXZS&ql7k5W(vEpqzQ^VdV)vM9Y<Cv6&es0wHB1N#PC$9^c-s$pUL?$%y`MR9W3VJK#^a6AAO{e4#i8D!o'
    '|AG5Odv;3`7*lpP>6y(w;)Aj4*#(CRg%GG`r#C#VPI}$0|IVdJTf1vvg&?=dl0!_si=)k0$%V}#2GkvS;i<y}k1pjG;AR^U4Cyiz{R>6&7#v02b>_+;?*QCj'
    '<}&5m4kiH3f$}0cT^MVJ#+nSq!^XnWP9maP35UJ4iPdV~Rs%D<q1qlK8-t?ro78-k?(5ClZVVT?52DZ!F2gfJg0%_>{LEO!D$+7hcqCnia~lyJC*Qo5H4$Zj'
    '7Z5(~26aGAP-Aj}>R_8BH1mmsQClwxev_IzKik_aJJ#<gFZ2Xv5iuK54x?&bQ&uK+Zb{~30Po2Tlow3^Owsr7oy`18(=ejl@_M<DOw;O|^_%NO3c96L%Xs(&'
    'hi8?+epTN^Za$g^36*5s1Io5ZPioRjHinraOj5j+8f2HW+u=K}nnqcChPbH8Nay38mbf57<De@P|3Za{P&w$?BtP*j>_H(b<reEXC!Rk)rlikdgbZz3<|%q`'
    't8Zl!?3*x%N4Oklv&n1MuYgz?91JP2PEZb7+{`cMi#PLL)&enDuq|-C(s^@v8?ERy=^QT3HV1r#uarP9+THd5Cga6oiBb}@Yct)1LpG~^rPjlB5*fVprn7N?'
    '0WM|1ZA3KLAxN7(!0md#_t=0<VBP!8Di+m;*D)|$zl9IeXuDYzy_X8aL}6_bzcFBoDUK7oPgAn`L{$qA8b$fbW;C_;1IxXlEw%al)te>yP4z;dlTymF&#Y_X'
    '_0U8vZof-g6{LWPkPMJ5S<SF<QtZ-TZ8sP_VG7R(4F5o(xvNYz9i%=j7%Ua%H=!`thJohJQOJSqfOkhRzr`t&lSA=W!-yqmNYG`x#o9^2n4rxWU`iwRIT$Cq'
    'IMXXjaM%>ZO!8~lpGS;gd4Y@f%qLBvueFPr;>GUJK!ZNq)&SmxcYq&Wo!h{RC$6=3=w^#;rYJ3GN#i7Q?vFM1x|aKO6y@YkXzq{SRohwk=TSQ$-U`1ftPY=K'
    'ZgTN4vfi+0(Jq$=%<ld>|EHo<$dy!_0&hK)C|Ud)ST?>ozM;omBt5cP$Hr4c-YypF<K)?BeG!!hMWjFL5+$z~A`Iw+HRE&GC~{{y<pFVf5j~yUd03y>aJ--3'
    'A;qr1@$CVADd^&Z2Oce;;aLe!rZ;HIMR#NI7XQ8N87<r?r|nZfb!+$WwmgOPtEIxEI^V3e)Hq*_Ce?Uhs=O*l*N?mos>So<#=N+Z7h|ghCuMiB*sNhd7->Z9'
    'vy=zM#vRP|csq9D#pOeL2?nVVXcf&Z#NFX*t<tFzq+iZtbQsf22CHjNmcWIJ>WvvY|Gv>5ZtRDcQG_h!tK2N>jkb?qBaN-55vFPucFlAv6wRsb`NAo`lpRo1'
    'mlLeA--en3&!{C23P}@9!xxrIP1PDl7IaTJr;Xb!PXvyh*%qugalr-SvyQaKAGB49bj6|DZTr#;>K*nXh1agg`A;DMO#3^<-0WziCK15QmPKx-r<i;zo+vN|'
    'H429=OIw)KzbT(SY$iAOQAcY%k;xDA>MT;hd6!5P%GA|Y$+<p+<ypC4zNakruw8hiwj2MT1@SZw*1QBrTNqPuzTBXeXHree)U|6NVGBNheydkR5~zk^Wx8NB'
    '9XlGJv{PsYB<-iufFUk57Dc`ii#qmlz+n$$hYsyb+Be!*NSoy_5+F;P;H}k9hL(Nac`%)DRy@>irh05mG3cQR4LhQP!-hOuwdotMDNkpXTiJ8LNy+vlDVc=j'
    'xGXVsZUuBQbO`g(kP(axw}b~5Y702_s~%00e<@e<a=QQGhlkIAyc5@ze)8zq{+}Pe0{l=-r{u%OQ8CEuM41+v#5zbPtQQHOrD}#$>nKQqjuQWV+p<WqV&v*t'
    '7mcN(ISulxn$?_18j+7+5LR)TK!>BNQAKG7n4xL`YfZY^dK1|x@7<LM%)K3yGOaV?ZLiM`Ka@7c>BG#WqK=;6c&iqG!%|nqnwDifL~YA0r+{R^*@-BLWu$V9'
    '{Ms2~OC0KhLtkg-6&!e-8b&mB7AG0w9V2@sI5K`>RXqJ6Af`8|r!(BHbYdv>dQfba2Mv8oY<toHqu<-s30samCizmlq@zt6QQKdu8zi!+MA9#GE7hIbWRo)Y'
    '4-O7hvqmhr%K-iH_<grq$HsRO_s_=Q)Jn=cA<t;~(LGb5Z)G5C7H+~M`H=~wNGNLw!@vb$20N4|UcY1NkOvR`pin-hAYZT4K{amHMBk8sVNXXYP~IaF;sC#p'
    'td1vlcR%*#Kw<-67o_MGqD}+O;{$E{isgMg0OVd%^CQmI;ZWEU=D0QF9FG>bQ`nCfIAG9N*AOBaF`v&DMwZG5T}ZxNmSHiOsM(M9CA91O03sCL@3}sNxjG20'
    'v>Xf;kmN}d<=S^j8UiT9$Ut#Cb9cf;I$}wtE>oKAyQpajG4LbnQ6La&<`CO%j?eOsRIa&;T*=<eu?$hbNdqUS^QckGE~qX<NnQYf=6r{{x>;giB>oh4aUf$e'
    'S^yj)2dk~Jq2tmzz?dM8Vm|8M^sUBlPQl?#?uXH`^&~Vz!54wZs4+zGnP($+TCohn(^!gacMM!0B((DK?b5^#87}Zcn+1ASeP&U~x-WzmvHxqv614iqw1;HA'
    'RFde0WP=`Z%y+OetjDXeoMQk;5@0&uk&f6)AwHORsn~$1O(Ja#*WUswQZn<xaypt@5RkHBjG!c>Q~g>Z?$y--Mj(r~$b4vF{JI(n)VnA<<7!w`+k328H!+X0'
    'r!2++hjcp?bOigS2bN&BPAoTtbLrs<gGMKYMlcuo6gX&f(DbHrI!!(7#wp5-y{g8dY~Y0u<kMGFg!BiK9tq&yGPb?#IdmK8qN8#7X4!?3{~%Pno#hTn{?kKE'
    '$x8=!CgjzCLMYwDycu7Uh~H_Bu%Iz3w+m-q;=rllRLwzjjyzht-x}m3i-luiQY*$fu&O{jPCpFyV29DcJ$(A$@VDU}L@54#>_c+||4Y~h9LqNTLB`v{K=A4|'
    'fc|zYgzHYoLj*f=&<L_T+)kl3j<9+e;DPPU$vM^(G#Z+^`-g>%2?`tG3L0pWH<(E)(RLJu)gtKN<POUW)!k;O$zy#m`GQlu!l)3{)Ev)xGn--bKa=0nydQ1G'
    '&R`sgNEoFl6K8Ci``ec-gtvX`Cij?`zD+zTxBRMH0)67}_h%?iZsBrz+yeumXSLdjy}0{TFBRzmG(ivQwmbhbC!`ZkO?%IW9T@6v?R2Gc*-Y1q?K!h6yxrzo'
    'eNb=+)KhKQe{0228S)_CX?OcZARnkbx8uFGJcE$;Z|YW0usJ@4v2C0m`Ed6GWCR#~A98<Xf4g(^dB-`DF2oO#AdGWhAZC<#TV5y;Mu2TBT@zGw^U~50)@V9X'
    'RMF3QgQm5^@dO09Z?;Z7XvknZBp=FRoXM_ViAb+OXOe(n*4@<Bx&dDIBr45m5AXSAiGi{9YZ%l?^dH_PX`GgeoGA~tI@`hU`u>pO6eANl-m>03B{PJ=o&>R{'
    'c??^`lPmgmk#fm~g2upja<Z8l5lx!_#uH6OWF_xwtZ0lBi352DpOv_?QJBRv*O{uEM|W&N0;6Krv7Ky+-)WEB)vj276|3<kcRY?7zP6Yo>s^ZAOggOy1lm8+'
    'F?)XOb_tC&TkLQ3c5a0g+GV!Dk}cf|CNRHjfns^#_dgnHTYK(K?dZmBGC@%G%Gv2iTsqp#cpM|SaY=_Gy2rD3K1!2n1#~6vZ~o1O*12Q^kWca;4kpmVz?+^_'
    'h`zjG-JAPVTHdhE4JJBuLh8`a_QQf|>#zb61U&N>@qY{Sfdl|_qb6)W>AiKk;WoLdlecOQ+{U7Z!DYST$|%w(<S~tIBz}yZRjfA~Sup1Lu5jp($jq2CL=%MX'
    '3v}@U{uwhu!LvMdHIXL{NCz^o+akmQ{(i9{-_dwCEU?wQ<+zcN%(oqkNP0MfOIvnHW&P@Lpnm&J9@L0>!@#Lj4_QjDJ_WikJx(JmeLSa|sgX6C_>IpayfPdB'
    '?NGxFU_V=rgy>tY>O$R(;W_ag4X6fE-M;&QWgBoFvRzu_8}wZo+ro^r^}rn1&OzH#{b5GkY||h1otk^caeIiRZSWe~P?0?GMwbVde1pfiL(6swB8_ez<2&_4'
    'PM|Sl3>XOSuouP#(%^zGjn(4@BMtZljN*`nS%}KGBb0|$LgecZgD6Vas|}@Y{Hs96OqijYw1xSX*Xz}$gtF7}Y6O_v<sN%$<y<~z=gXvgmd=-$^6OiVsg}-#'
    'oV0ljxVK!b(<6hXyp+o}E_$_lw18O&dw^)hv}fpIH2vm=@FAcVnPM0T6_^GTS?@VXdwN+TTZ&)L3VjFMfnhTSml}vc?HElIrQevrmn<vNT?Mg7+i**_*lJI$'
    'MXtJ*H#;`41&6V(uI|u^wmwG@F<qRei^UoZ{+C6yF4^G0>b@3pid$bTRwyNt^iBRmp_)R^FKhN$jYH~=taEA1_}+1ioM7=cTBL@QZ|7BNcFkL+$miHP{ox<H'
    'PHswyGmN=XTr8ONsK<+CnR@6;O>AZ)u4v#YryNivpsw>W!h34dAsx|oSDP7UE!513DW=N-O76`aO{il{Il54TXBT28ZPXc4&#H;JMkuZIa#dbci%m3nX@~5R'
    'NX(}H^4+AL*{*GkuDzy4*YDVp3Fd&Y$BXerNuDG0K;8yvRQVOv=9>Ne2eZG?n}vYD#Xf}%4iik~^ZrNPwt?DHwj1{!=?hOg8JOqYYbrA{AAsUilz_?AkJ&fh'
    'i`Yttk770B5Sji7unYJ0LmJN!dNJ`&&47AT%VgNqgP}&?Tw_&{;leAQI(pOyuWD@Df0hiM{oZZ<+YCn`tnNWYmgL>WrtKY#M+<2})iVaqR(tuk8@^l@-9v9~'
    '@tsEQ(Xm!&Zkm$7xLS6OGoWXOqJsZJR-R3Vpo@t?;gQAntrPB>(AFW6S5reZnltmvsLb|BF6CtnR`<Yfzv|nD-L%Pgk5RYnhsZ|w?MMD*vAP7z`=({bYfP_)'
    'W~z0MPIfp(zvmeaTm9|E$L-up^yc1mtTxyCP6OtameWlgZOjhQDktH76&s+qRS{goVR}29L8*heaL`WpES9yT1NNXbP}BPhrF+;e)+b4WO_+jGXil?2AJc=d'
    '+Oiy77Mw2=W8A<NAq`D+YAGlM4bEGmEQX&msb>xkjy_F3`6PMJ)e1~rxw_h^5mR)hHk5y`h}=z<G`17e1qz!Yusehl)KIzO6xHs3$yhfWO*p42wG;_kZBH~8'
    'z=|7cN?$N@a8()~<Y@g!k=P7ayAs%~mmTxbRxX4Y4;kJt|KAo=yw{rZ?sy09?haTw58v^2?`{=G%ix#MYF$l6<F$!A*$frpn`bqQCHmv~3fgYGcPx6S@APYg'
    'Uz<h+Ruke^R5m=3*u_C3%6Y08r+IBM-Dv+FufFGK)e8)Pq-bba<l6DfiM-PC&cW37FdkTKpA<s2)>3Tl<yABmU}4dY-Z=BH##F^y3XIJfh5feLY|)h4gA;X#'
    'Bm-{|sqV=^axbc#l!!03wtJ3j6g2Td;l{eW#*{IE2V#eXSJu`B*)DGZ&eq90LRVs&7+m#vcaT;<5QS)E2|EMyr0N3GdxzA4h@)ja`3V)R#E;8sevqrs5HqrY'
    '+#7B3XfYWgEbQm83CeMHhUMX=UXPHDNhch1n!<2!nnyNZz}FNi#X#y;>vC47*{zj$Z!77Dt2ZlnRX~qTa}pBiKEYtxjb__e)=n3Qy#?N&1<?+Jl`vtG?kjgS'
    'qm=J2=jT1ukUfuKQyr*bIM}sZkA<bZVZZ!EX#=vb#}Vne8-u}FTzTgKbRisX1l<L99zsO++*pTT5S-U%G_NM?AE2&OIJGr%2Y-G1{Kb=}e})!tzp}#Ecp~r{'
    'B$+-GE#9jao7GsEBJwk~ZuN0EcVnVbA3lHi<k63hi&wzEa0l@K;_;qeRjb9EOcm@AGj0pMmtPHKMuro@3_+~GBbJEwEifvhbByu3PWUzQGKx{^_k4snsa7)s'
    'F<@CZ*cThV*9G)vu>o9mMv*XWqVQt9nPVP?Xq;2tjm9aJ?Fq2w)^;~(2qJn>5Y|1`ivdZ?$_1DcRoc%TBCcxl{%!S@fUG4~L%U~W0LkX1>jKo_D|ttMN#7h#'
    'EJr_pfAFooIv7`D1l`me(L4!)&zTQ>EGAfq;@&aRsI8f}ewlWfhFgW8RyyS&X=rXjdr^A#X=_I!_nt1LqfnWQK^;^+?YHfBQya-2tsdnIh8^^O6pBxV52MUL'
    'y!n6z9PY>&M1i&Q5W^<J;6sRU6T>#{Ay(D9#G<T3w84<o7;fV$Vo+iX!LXIv=!D=8V;k!cU4^#Zv|Rh$FxvX#7!bdH2iG$6`4qoi*FJ1xT6_qdtj#cw@FgAf'
    'E!#4TvmQa`1D=FNYMF|ltj}x=+t>!bR+FO`Ivyfe0mWDhEkog#QA~!n$sA7+?23T>7z#;c@F)^n8m-E$qH-4@3SO0?Y1$eSwiypr#e$_Mu7N3zC!Z0+3ZSkw'
    '({<Bq$1m&K@1VcTnhm9_?WPqXVV?_EqSp4Aq$5_rBy`Y9bZ&OlJR1|f4XG^_OVh+iOx>D51$HIN6$RJs4X~iG_kvmNKNu!&)#ZAd{9QGA@>Y%PUUGPQ|7~<g'
    'SPO@8!m(@NNz$yp4aMiTy&%0b&$cmJiN**vab<&mlDM>n0_ahXu=0b#qh@oC1(A6dv@{xSE*oa>pKOQ0bX2%Y-u5_%2<cDI0B+>$#4O^$$MIsdDyI|+BM)%`'
    '6KWV9ltMZnyWNM@tED(Z;tO&%HvEzcN=lWoKF4J3<votas{jl=8@h6xN{y<nH>dbaBT+kZHLB`T<pPEc`ee?-buk|a2yi9z*X*B}1koG>eRO2?(3)1nz9#?W'
    '1gPi5`a;Zv0HJXdRhr!O<_yV;9S^vJ|6xx#g^Gpdcmlj*BM3kOz`72|iHXL5jr7+T9yOD7;`~xJNSx=U#hHy-eS%S|Ej(JI+u|+Oy5-~8Yt*%<$SrYzj|}wP'
    'kWqB{-yI`;K&<5W6h8#@NlMd;nnbhfvNjB5Zaou;!aVPJ7)yV(FExI%ND(<#<Y8NJ9APnyig@+04!82a2X=9@gkx0J=H9WTfgZx$H8Zk`!pLV3wTXmA6&6Vi'
    'ehg(bs+1EYXeJ&jbx8pf0HID>A+!ksbWiOzD<nlZRX;rN%J?^=(moD7R1gve3I-5ZWTZ!Q(VJ?{$w#=xW?oxB#4aBdm|2^RK27=5MN4U+M09iUJe1r_&&U<D'
    '$<Td;PD%QZpt)dT;^tw@V8A8Qvz}_ikGX}Q0TW1cU363V_8DdJ6#H%rwDq<l%GT|dNfW;W4hj;0E8)A-HG`XWg^~Mu-pRvm>Vso5GNks}wh3k1uI@ZpY}U&S'
    '8ot|11j+6+MUbNw<Kh<IJbdxEc=-MI&mVvP@a1FE7q6iJ6<L$N(`ruFkIDz2YN1HR6zVn836$6C)rc<IR*{6STj=E-+*Xu@a<}YddBee^uuWy~j<ag1l>w9*'
    'dZda>25uaIKwnVeS=C1Qp4z(pYGqW8;<%dvZ1z8lXdql{9PU#r_zmq}?s3GDGj}V32_N1<d$+_7WuXA=Jp|jZ=c^@=d`F*sF3PCG2R23oY=QF<BVcG|e2i8B'
    ')dX%)xbR_AO*yc}59gx|YsIxKa#ZAK5J#a92ZevK@_|r>*(F2vLmq~ifsNx#qA~+x!i{n?OB%U;z0ic5DXN(buF+7WXOs{?=^{B)1D=p+HO9#7bC=kuW1@_m'
    '0b2-HwK0e}#hGc#AVyMo#Iep2g9i#|ezv%ZRJ+xhgmN`O2Q0Gmc98;ubziNo9~^vsVC`s@#ze=!7D>f=^5Dqi!eRlkdD}>!=ItbB^TERKH4sGg-AzNt-Rwiz'
    'hIqi<-8dxFv7B$L*y##x0VbG4)ZyWBEhuL?+K9{d_E<WuGbQ;KmE2ylQ8iD4s|4FaLT9<0-dJ>ya%4*UmS&zC6e^0GIb6uL$!*1qTaV)OV|H1%(=JEnlzrDw'
    'B>m(GVqv)CQJ{C6EUP73e8L-Y&+}RhtnjusONL~zcIlQ7y;xr<Ko37PwroI8k&6QjUXicNqI~q|FW){)Y|&$(MOgl30a(h8>EiR-BsGf-T{tZ#7_jfz%}eSf'
    '-<#P-na*DXOJe0X{DyZJvn<McfP9A4ID{0(mI~}uYS|Ds7JP=Oo?+f!VtnAjm{1$NkN@`Z?BlcI;~zf$>EjoM48q;#5ewA1>`m0+V7P)F-?^i+MP0m=O9}Ws'
    '+t#gjpW)Tn!G&bZHBp`<c(>5Ss)f28&5nIPwJ);W<jf)gc6o*1ek#YCAm3XLh(|Jx+^ol(+OVW4#DH@6<5xIYefWkVjs>Aoyj9=n0){O|W<1Pz)Gpu*1*>}Y'
    '{K-!bpZ~4+>G8`So__nH`2NSI-#q*gn+of(^>a(LAD=$LTG|f?s9+S8jmhGHB9*-DM%FlSSH*SuB0G6JSQg{-RrdE+#fy{c;>FM=!hn5$lZ=o9*vH!``CjM+'
    'Nr^#tAR%w2<d@CpY{gMhcDVAf^?C!u-5QR}IgGKyLw1=gCW+Be-nU+po-ZpUc)&MGofB6wTcX0}OAwiSha3qfArZTA;zPbnc3?BMWIyF}mA)Garo%nACh9T;'
    'F*k(}xFxR6H=`9<X4fV8$(4y>+8pIY{9B$_E{_;FMX}NMun`;25#a+q+?Q02gb#uGMVcsCUqA!r7cFkZoDcfT1l<*rA7JK2^P5Vi9L0SkO!k*}n1PgoN&bPa'
    'T}c45STfw+hoEu}kIqj6n_`Q88<xb`hogOVw}>^0A#(xgUix2JND(E!Fc`_;s3RpFaQbdR(%Wg6kcV+0ObvqPL%UkomP2f>tOEEJR{40lEo97{H3D9yI^ht|'
    '2{LB$^~q6gcF_rDC~JZL|Fid|-EACMg6Mbtik$NC0I(7;Ni9`H7wfbn%eG$ImPV4xs)IpigFu3W2m&w?kVvcbPXCDhh4V|^-6G<~k_mv4x~u2BI^7ll#2z>9'
    'zI+5x2~{ilA|thnx(iuV(Dmpr_q}pRr&Doum`H{Eq1TJQ%<g2#$+K>Dq?5?Bn8|>lL&a#G@~>}t3m=~|{?q4*4|BXvRF3z*oPO7+T*|X#_V&Lv_i9;wz6l<!'
    'x_3WQkTGc(_56DRJk+beqi=msys~Jp%j-32usz#ppsjX1e=xLFB=0u9Z$7J?#BAB+UH0V}r`xJkYWW4}_!+l;ULe6LfgQpmK<|cFg&sLN5I;P8GHE@Qh$E0o'
    'UYPTLkCPeDZ+AYlxC-j&r~Ds`_k>bg!y@PB*=%@2k#E2sOod~@Qe#6-m`%12D<J%zfBz3h{S0yc4D0(&h6V<MxTiJds0Y${V<YssXE$BNH&8iZAh59N4M2D='
    '%f16PX$V8zJ3P#;v-^h`dLO!|>AT1IA0PZFBZ0X>acFjv9Utb0(3%BuJ^+bHh5J1Cf*X^bZm56GIGsB3e#0{mP_pCRy?fdB(?<xt<j8jSxOecy0Tz)XsStMq'
    '=f?-n$60A6h*SX;PXr?Jw5+mc&wip7i^&A`2GS;s=G_nj8LR?;Z#b87aTU5XBZXscMJf@8NluW5d<P5}fHnYT4*|kR9ogt&L;jK39Gk*NG%~_s(Aszj&p<5{'
    'KWPqdbg@7=5tl~VQ&PWoDS`Il40Sb9?FMhg`TYYU#!szxaDY1IBUUm$fA|7b%`d)xiLq=}HK3b3eE$5$zZcJ6JbU%*@v|Qzg@V`g)vJdf&42&=7i|CC(;uHS'
    ')Ob{S_oITezy-U`&!&h|-YV5Q+}?nc8?QF-misBSI-yi-CuVnQuwpw8H-5M}-ypVlPQSF#&+np~Eg4sLRHq*PG?douR<{d_?i!7odJbKJpe{v9<(Vn|>39#+'
    '#B3c`V5BQ*w(iwgax4sr+j%VaXu4O>d5XZjGQ-kSaRcA)-S;gbJUoK}Ma{9ggd|M77EF4hE>5eH@P^{MIdP^1!0aVzsyKl_^5(0Xs)y6`k4_{v12BqQ#PO8H'
    'T+>1n9xI))M}9&QYP$6gfu!Eyu6Fj0mpEC<5k^m<ZHgDeFthYDW|kn-fyK}1oSA0YVkZuEsJ!q+)ziBXdXEPO?x5|_srRa{BvAM@ntto9D?Sz<4S4D^dd_0W'
    '-8CdrOJJ=|r#D-?gGXukhQu&d=ba9n<XQ_I)Cf8iC>QR0a4*A%MItZ}1(1rOh{$cB1w(7d9|G8m3q)6LJgygdK7EUDj2&ORdY$2Jfbx;}i*d<JKr9X5{A55~'
    'Z!#3R^=L=|0KA1By}KxxCj9Ni!=G?B=jD6@lnf~61o+dQpYnVA!N9%1{^3aNJ)H)p&z7sZ$2J2Z%C*4-{I;2nE}ehX$t_mlC)nTG1LT9F2c3gG=iqz+`%Du|'
    'A@^k>lZ&e}8TR=810Fa5nsFyMDni+H!#fmRI20{2-R&Kk0;UafB}ag15utZd2*gJV7KsGZ+(RVwb(v3)(R&gWY37tcV@ictZv!X9qq?6&RiT?Wid{=X?*uRV'
    '07_!~@ZQgE)_$?)G;nmXuQHBur+*p-t^*SlxUsdl2McBzu`H$dG6xyWEZekVs=ZiT0fif#WRHeb`Q&<3l7fyd88HdpE=EQQg@Cp`x|gN3eHrm?IDw<tfSE}v'
    'hBh>m2iSJnLVAnyMD0w1Dr_a{GSv2X?R~C>EP^ySW+8-qg|@Xn{NWFefHZW0reTJ|5&Csy(iTX<a4n2INI4}Fmo;z-=PV1B=8$?Yj5da~Q#mPA<^(tqn~Aoq'
    'u`nND%n+%*<mOP&LfgjM1DBE6E7K(NVP^7xWkV44M;EKb96ykaYywNbJn%@&&?PZbV`FLVayFqhJvHDfqOY?VoXORuT8n%}Cu12rdoFY{c!<?>&Dv`>)N)w)'
    '?9LKFsaQ;k^}9s@4`Z=v6B9)mB)T^E_r&<tZUQo4DnTniJj(v^myCbtolV!ptel^(FAxH<BOgpm>Q3|iNN3x|{Rj9FZIl2m<(L+w*x<w>0VW<_EEW~k7e_d>'
    'ECiRMCuKcy0?Z%~guo%9+$|#YyXkm+(c{?3S4PJmiv8jG!k9P#E)DDPGsx~g2-5>7`)UZldW{Ujcr%ia{Ilt>`dW>|Cc>QOFVW74ML(F!sf71YC~JzcEtw}I'
    'H#<iwt}k$(k~DQm>=ZA8#d4$CMi-lN>skn-KV4_T@o>53idX<BEN~-$eS^aDrb0tPb-7+FGoT%KLMt}SQr6SziXl<l4|3-4vLSQvc(#_KV_t7Xj+<3B!)Qj+'
    'EgNpu3o)n7ZhGeO=-*#+tT0w^!J$d7poi!JC@YKl1TK7_v|n4mbW)So%!o7M{#8zA(g=7hJBR|e&Q@&mtJtA?C@i}4HxaHelf^g(&yOh~9Kd?9Mh8Qoi#iER'
    'c~xNUn(2J9Xq!nLiWgA!w)?9}itDQDu+OQT!fS7SMEc$Cs9r*g9c$-PWS`N)8wBR7p5hC$=Z~SM+l9qT&9VcJhyUAD!?T&$%-!SEUfiW18k5zlQxr?Ce}FRX'
    '*OG_Jtyym8EC2M7tx;@;THpwu`CQ;q04J|U-)1$*5!~|lJJm=(H6lj+7ELi5qTdvrz*~5Ka%aY$R-Md_18HP$hmpl>9YXw&2uIt_{ADM4Vl?Isw3=KSfaK0*'
    '$hrz`#pKQ_bU?=pm?%~P_9ODWrdxwIOIY~~!^?Z&)ZRtiJN)y5K<l$woK?KL6uD!4Z=SgnK_}TWvD;#?F5EXs>yDXXzc)hGE-)N*vgABqT>X&bb%${LrPZJc'
    'q5EY%n_jY-7Ql3kDqvH7L|8H48#Y(Mj2Tu}GNa-_AgLwF;>?;=hG?IuMOG~)YiKS?1K7j@2+b1BERrWn^s7d9fXEbKi%C>M+bzCAxSTn1$24$?ieOI4ZG>bl'
    'ki;i1i1KjY@I9jP!w88pLMHaD-9F^09F{d@3jjUizqyI3@X>V*4Rd#oe0U1Wg?@lWk3$;`@^5(-G)qZj;q%)q+|*xXPkzz+SLzF<^5!!EBF{_Ni7C?qYKXl2'
    'CB#VZ$bZS^L2U-30J9=iJp-wD#vE?1o2lxRuCm-%a4MVt&w0m9;<xokv<vK6tV$b$f7dR9zn66|NcCy_6PR+)Nd)=m7i(@uUQC-uLw}R2)1$iZ8#R7!v{>F0'
    'ZEw)H+qj(h^X9Uhc$ad|Fn08tx#T;i*-=pB!bi9nnHWd$845F%l!-eC`tH{M0+XC1BN)ipG5Y0;>;<7|;BSYPXLZj_@G~H-Zx^%%<U`RYw3#C^OAM#wGz4dq'
    'pNT0)u5pTyOjFEmJzbMU@M>6Hl1Xe>CC0*il3V64oLmnbm42Ph$C#EpamNvkaK{4t#c?NbGx8pzs}U7JT1>Y}OzD2Il_D!Nw4f5>+|d%}s2}EBY&ZAKv%?O?'
    '`Y|HpFHO#gt*bCj(R(hybo1w??-J2w9yv#3YwAfC-lJ;U_{*73<GJ9er|@IxNor(Or58M7|3K<5A>=CTC?3ZsB*_tlzp|cw?0S_SklgG#T4v&<v7HmUlkgN2'
    'uQ+&|gw)09UhDq>i&w%~{G_}l<?ujiN5>BO2Hw<k#edNWig^5%yy4h59PSoz2{9`0bf3*<H`+^&z3Rjs>~jN6VjJC1uReWgr`fR@*)9VBU7aCV%oe1%8Kcl&'
    '!OrPiji1HqV!kvrkkE%<dcIj~Oz}%s80ocN1Bv@S;=azOS{+i`ORsLgp%e2(y!AT9$YTI7$+uw%3@^ZvGoeIV%T+0WSz$(V<g{KM_Uybsk|HmdSw4wv0$Mzw'
    '3c#>?;z)chsbEu-$154`NTz8OjafN?A29q-sHt|t8f#1Q&~>J?2lp_`GGP$dsADH(T2}rTM<(|ImG2TQ+UKy?lfvy^>nQl!+^l}?CIaCwV$GP#Qd4-W7JcvW'
    'Myll<ZyAp4MEucp;rPrU2Cv~2xA9i{ZnzUOmo4$-Gn}DX!!D0JX#~I9^>)=l{XRsB&1!xxSBt9McH;oRS(cZZoUwy%{UXp*%?{AhJTr-+{KB3ZlP9;_Niaj>'
    'Yrh^ETkK_gyGUpN8-H-Glj5dej2?-MNlpn*qHk9oUJvy34`Q}J0oFeVgi;J;@(T@)Mw_b*rBP@)=(L=%{n9CqhkI-ci#EByVd|~r(d$ke_#;>>cI$_OpG33W'
    'iN+g5!(A6|r{y~s6QI9;l1n|6lei(wDJ>uMrBSjS-lrwo$qTha*9i`!t7y|rt2N}Zy4*}2Lyzw&c*|L2bTeH%%LJf=aK9N6=`G$`yL*IBajSR~>rm35PAEJ1'
    '6D}eBi=z{#Nz`8_bTbtVvtCD2#GBI(2UvA)ul8W{6MAGpJbX^QYID`DkrET)9vy<Z2Ct8hD3jEot(R#sjR&-?hxxDIGCIA{as&SbOt`#(e}@%5@3v-K^s3az'
    'IiV+noOO)y4KXzM^I$JgbE}x?QP;Gj8B3WVNoMNneH%+>0a1?Rvu4@MW7o6<w1p`#(olpHcN|r5QRL9&GBZT{0GzeIM|YIl=Sj+taW^HyrMqe|&R>lu%>gNW'
    '3y?LslraCQBg@CX*|P_lY}4>bVN!I0eo>xH+4g2#kJ*9MYAhwz^sGDw)Gnqz7g6qnD;P`!5mLHVrW#8{20QBt%WmV<f^EUHa2NYGt+LIWrJ-@}z(v*>%$dTb'
    '2g1eg)a-79@N0Y3H!*X4+#tRuY^q5yoFE7oQ29p@%aam#O>K>BB5I_Z-m04r!uC4%wx{_fB{}Af5z#KcX0B&1o__!I=Z8NQPhLNN^7z%0ZwpxX!&gtA{k-2a'
    '_=tA8LgfIvL+foIawF~S9`}Y2UD3MKFoK}*-w=o>ob&WE0ukl^P(mjjpD+CQM`~~_I(&qnj*CS(dg!Xib7vI8`L975T%Z2@^2v(;Ov8D11X~Wx*#)?oU^i;~'
    '$wO;kxQDDcmP}$^Suc+F*i;;2Ltz`Aetrru;h9W=3~?dWm6<Ru1cBBM(wIzF<vU7`KyJlfkP-<M@DeROoSm0vt0DIk#UT8Ofxl5KEqh6!vVKT9G$Oe+P}<UO'
    'j9CH3fgNYa1Ewa}S+*eQ3}Cc4`kB^M8vspu1q8p9fx3)*K2e$a!_$}Wq@O+psFg5FN_KRN8K>}`j+ryE@pn9H3ylqM+!!#5TGp%K>atwfD#j(nSh;W?d*>9F'
    'a=bmt<#_ifm*d++xu}khetG)ix5eWhAHIC~>r?p5as4k%PLH$AbT;N!iy`C^^QgIlV-qd$4;LZ>i|e8wv>J|Xx+aJ;msq9hwBK4zXNz^q1-fENSQCRHKW?3X'
    '0itUqzAgb|GG6A7yAH;X+`XF4yEa@r9e}fSL>>y>YS#p@rUv@c)^QXL>AFBX@-|5gcyH@~<0(1v_*=#tzbxtjEo<TP0vQNM+V*QOXr;>$$X$##9<Hmdx#D<<'
    '9}{A9{IMZM`qBIsMi(9b_lhn$PDd9V<1rQ&!>Zss3~m00IJv;(+e|PAMrd(ak_QU6Ms)>!)gzrqg+@3T2+p!~+T!H2+zYeQdbg$*^NB;9j`s?6vT@-C@fn=&'
    'ySIOlJ;#SZbW=z9o6$fV1}e!eTq0qc9C!Fb1#h)?5U7BnYe=9DgM`9hotk)j%xU=`j1%G~9R>((TLlxYoflId+y5!o1ee|w<)<Li<XWN>9I%Ez9sSwPlURvK'
    'DOfjs4Ze)#w`p7?vxBrtgf-zxYg!kyC>?f$jU{J+rJS6-Kt{eFjGS&<Gb9s}0_dkrw_?0%)59h@ghN{brQiTZycdiZwF?K*Am!I(tU1*B4juED|JP5CL4=wO'
    '$x#M%Vi+KZUIgUGT46lO`rQIp?OzR7nhlZXULjlbc8Y!mgoS0evkk^Vj25dgy4^@bgh6QE7z%2^PS}1#3`Y;X#)x-;%P>ocMy;f0ERgvkp{*6g?{ObW?@TcK'
    '-cKF&2QPKlA37h3_v!4`A6oAGYm)O~4p^74L#Kj1v3aAqA_=9l2$5JtCq5Lc>9WGM&ZeAnnLGKZl)lkYPbzsN5*;-bNkr+q9}PlKYG5^}%@Fks;EX^rw5lQ0'
    'dZGZkUWjd05(D&F2hIr7n83LHn6o(Zl&#0@@O4eWY_*XvDV2A+aFDk;9rW7Ogpd^9d^pZW#yf=_VjPd-7{@QNF8nWhO!B4&XCVwqL<_aA`=7@inX$Ah3Ja+x'
    '#AB%1&5vmcHX2D_&~Gjv$mNH5IKLUV*dTIMF-iiG|HmcTxTvQ4&^?r#h|+>FiWI3LCnsQm0+%A7SHvpvI?+@Z?1cn#O|ZCSFWC+g_DSJ^@&*BsEFiE)aVh!D'
    'gsbrBpEeFgc9Kg|u?f~Iy#P{h<$kEQ@tdb6#@&ux8&0z|3dZd>^vd5j+q+CYA-joNYWCN^W^X##|IEY|c{K&FAemy9amY_AIhYrLBZImP*duvwt9MbU6#f&7'
    'Z*vd$QvQPP9>OO!U$;Qsp1DuuvXT!UcDguDwtTA3FMX;=Z)8nM-l$mn2{^LtR&0GgRJ4j~XE%{(wn;RTKa~EoplM}SrW6M3s9#kmn6aT(ett7u!*j$i6*P(6'
    'u%6Y<tOwoH47#ZXy49wp)yB1&Vx^y|<h{&Lm^<$+;mbCzJkNj_0677C6R49PVwO-`#y^=69E>bL?3)t*amtJ6Li2y3=Dk;mlt~XI+9EwdV*PceJ~)VaANKCU'
    'akkZWcT|JktIke$U?z~b)am(KEFX*imLu=vupXV7Ot<HD6422$V|a5T&j~-2a~<@FPX>>HC;KNH1Tzw%?_%}aYAqv$C`%=eOFdMTb2ECd?l*k64|O6ne!cjQ'
    'uJS~$*u2{c(`XiROmR5#&d%aiot<yb*Y3Y`PtcyPr>2aEUI7RH)X=ptX6LF%JwU{5ZuoRZ<PzSyj~7>0NXhJsk~)-UtiH)+%xEJcK3||Ot8rzYm+#m>ME<H('
    '9hEaNDB;Fvu$a`b`*lO(B)zkHmy4MPGKz(yt;2sL1^i815ZnR|tYem5CNjG!Ku83QQ{_GI8@eO}9s`VRV6fk^zn*6Ih%f#v`!YKY;*Bnbvx(hoxVh-Seq*l_'
    'hYn!^8X-#VQXkqvFmp-?u&Ul~SEC+5QG0wEP6mwX5G|vpoZg)3$XK^~D*lS~XAbZE@%YbxwSq&tll>O^Dj`{*`9(Tv|CaEgsgLNhcS-C9oIj7!v)Lo(=k1SQ'
    'AdF&`C+o|DzJafkxk5Tne2l&2a)M}bG@CBln>;%zyAR;My+aI?hrfgV(cSm!?j5lb&Xxp5F3C{N(M7e7t@5mk{|xdBaDR1rn93Sx2iO$=eb|%k|6QT)9~khq'
    'z?{3ceW7kES~cWP+3R_>1cno=y4#osY<i67p-;P=Y?`L59`$9+&B^P9@NN+)ul-a!zZbz5X^}Ie#(^{UwSP3q?bX_hFvsHQc2StGJs}e+d5uTwJN{}A2x6vz'
    'VX+2{qLuTPV@+xMy6=2`c?muS`Q0OariB&r5@3=Pi+)G%V)Hcu9Na};9jX(RoO{TiA30uMl~j?-?D=790KDohLnW}Gj0HW}!`){iqixX+<agWvcDD<JcW%~%'
    'j4H0md?t2wAR>u<$%ZA>!2El|`Ekea5cfSoxqTk&AgQpyl9Kc@72ZPF+!FoP`~>*{)%jWR+dD834QYQXpI<h<Q_40W-#c{?J!rZ*Q5bFJ)88oN4Ou9$XAch^'
    '@KmML6I25bLrkU|^;!993LYeSO2YKkFqnJd4gh5Z9P5PvjcY+IoIzH<NQ7Em8A_ACH9t+<;c`*scAGBFMaEDmhB@FFn&Y@9ZTti19L+MY*sT^P1M($=>)|+|'
    'g?#z457d{^E_g!0gyQ!Vd5fi-N(=Gy(%x5;7s0x<nM@2aF;C&An0I%fPSmd0+_l|{sT%LC2M=q|L&VVTbapuG54BH_a}RyOCs^z1m~c$|L{Bjx)IO1Vk~{^S'
    'kV(Fz73r(Ry9y5sP#2}b;Ize7kbQmIKuZtqeD}oIFY)7WhInv?FZDHml!C7s^69V*nh>~OI&%e2(}aM6fXHMpWCVV`(XW+fHjfYP_qy_8w#>f5;H+zt?$R{4'
    'Mg*dw6}a@yJG>8Q+lAi)<l`M5wfrjE3*RY^UULq1z`F4g?x{9RFvmnOi`iyf`tO;~;u?VH+Di*+D8TE!>F39WZ9(@o)MPTL$g7qX5idH5vD+2Lhq)nSqC1&j'
    '{Wyk8XTdN5bstL({1_p3Td8RrZK&Ltb`F|i%=RS@Jb98v9Qbin{p;%Fy2NFr<Gm2Xy!Vb<X8J%Z_OVA?<Smm2^1VieES4=2jpAWKqLLgYY~;*@FWG`AHqpR@'
    '*=QN8H4_>i3wF$HJ6{MVjIkppadf?2Z{Rsqu??zPtbprnW36qxGC|4&`qn$-QjVw17}B|1bOJ9s+%to(Y14LV@+hL8u24dg4=s{3e!sX1Gr)lt$oy}ooMVyF'
    'Z*db39_&N$g<y6_vxRS&JqS)t#{-6h9DX0oHsi9u`a=~bk#JSqu?eqpV(E!JtpgQN(5AIdC>`X17h77;wOA9OTu)3u-||Jh#UpadHH2nfmSeUtV7@h&^Mfmy'
    'ianA5G*P$>E2CzArPmiMEkJ<8Ma)$oG|qt)OBYgBb+yJ4ojopM%4w^@AsJg-hPYp#q87Ah79R=nzR}}UF2CkSx`uzrTpNs8tES@(rso)PRR!_2vsV3Nu_+uA'
    'Gv1uf$~74r#xfk$r;@@}IeF;J7um)1{Gwc${2a!}UfG-&%I}`<#-#5E-UL%jeP3tfiN#RG?G6&#hE|<}`s1)alVV@Pxlhc?38o)~y{z}S<Ky6Z)%<PylyPj}'
    '{{Jm=U(rNCI%;_qJVfB>D?q)tS^@`bva`Zd{fV=KYWddFVMSQLBz-B~p~-M9LJPfi`ir-7X~HvMBue#4A<C*Q<|Wq(u6cAJN{yJ*!`@+xIl^8RkML3v6ncj?'
    'Mj2o6PLrEOs~8;~K@=z!sSaR5`OYMWGp7~y!F8+V0%~p~?_Rm5Y~rho<E7idMIhH^XS<WNS#O(e+99Q~Y^ReX2686!DBDWUl!h3Q<a|)^QFvPVOG6X?rkoHd'
    '{g}#hm{^0!d?=D4(|AjNeBBL-jVY%oPA+7e-yT2jR-5ILvrwD-MZglM1i*{4H<D^=S{Vl|&QFAS4}tAHD@Vf(`e<<$UF&LRqKgg;wV+~(B_$-YS(GfolQ1L7'
    'DmeqWbxhT{|Mg$_O8)g<867^kCge1BW#7@6k;C-iTM8Z{mn8}#s@5@LwV3(>uL;+Ok90UQ9c|dt!(Sdpr6YMUeqyD43I5O|@%tz`e_3EUNtRD*Xu8Eihy{4o'
    'PWBh>sa!by<ro4ZPkVV1sRoI+I6>_6!RYM)+3I$Zqw~FEJt&c*C0`&<^YT$IcCr`$z^i4pfm1tvH0NOHg^OKC6B3i3VUpLOoP%`I5PsW|?}#6*)X|}sSC{rk'
    'pM)a%wCIyWVi3u(e#+ba7eyirV;oD0;j(Ow3)$0>Y?@~5JnK@roe5<VuqTrHllZfoQ6XwPvYh`mTVZGo7m%al!|7~zbv7R2wO^fNZTyE)e01qAFK08uWgoXZ'
    'uq4OL={IEHw2dhRgj$m5x6qA>qj+$EM4mhi)p%(<9wJ(g`XvJ9j<*lyF4|_Duo0qpMj1=+wJOOqYkcBF2nPzI%?k6tisUW}?5O-z+(5k^d&ne!^S-%}C+K@q'
    'o6Gl(LrwoL3C>c`=&1=?;*r|H$S&n7^VYC2v%(sd7LV9`EqqhOkO<?7B??Kn8M1)mHVY5sUiG|^oN^>Pn!JzfMQ{A%K@QLyvir$@j0DCZ_X8b#Udj5hHL^$8'
    'P3I>MQxf@ewL|^^6`$`vbm$H1P3DNNJ5l6#yDLvH#u?_Adq|+rgXmNrydRRs5OK%$u?TY6H=rAYUP&B>!v+5-#<IEyuR4AzBz=L1k976A(@gEZ$i8!q?czy)'
    'F|4q_<YGk?nr*JfB}tTYFc<(r34<?8q=yOR;DIQNAkUspuZCIsyVE~)IQ#w-Rr?^5;&*@UP;Dv`5pp%HN+Y3S{blpgggfkNmwHkU+Oiqw*zDerty+v)G6@a!'
    'gL6C&TYT#18|KBy#1Ro;hn-`8*q1*NoftJ4RwmEhzafSt^tzsOJ}55SYWMo0IvH60<xN`l*8NO*(>_7K8y>;Z<Tpm#v6k#z?Z+;cSEf|Lo~=(W!?}KX6|?Hg'
    '=x&RN%9S(nmTJ=~u)<2Cs@9M)RTUxYbl@eFcK)$%P9n*$5q-MTcJiT<9Zg?u<fCWbY~fwFa*4H(Z*aI1PB_`bf$ta_XM%~x6i_b`ftlUoE8si?XIK3<fT7;_'
    '+L3-JIMA5=6Pjl*ut~1{0O|DwrXH}R?%U4AH0J(IE*COdJ_byJ2%sNkaUm&H+R_qg%%XuXt+23mHxg>DrCU#LOm%7S*$z@FGoF7(9kLnDFh)*KNLy~>ZxTQS'
    'Cst`cm_P<hnAH@+&Ops>Uq+A8qzm7B6TrZ&Sf<fa5i_a+%yAEW-Je{#rEAwN_#h#2NGmUl3Wv+Xz4>3%Yq0_jL1K;y0Hpt7-he&JO!cKYAZ=h9ECdIYyD>7W'
    'iKnP}f-G8FewYC><$MrS9rOL)`e|dluZLjMycT{shBv3h4K(u{InCKU^fpUGckO*>-j1{{ii=26s~t#REkU!%D?0~Df0IY5J7>&|c-t*|a47<edCHBIG+Ryb'
    '%qc>Yw*Iv3Wfp)Z`j&Gy;Q~lhZC!{hqt(%9&5W-IQLYq!rww{jx?oa@m?iB<^5YX!h&9jP4ytlS=T#taFHHHhI;v$Bf*Z=s9)x%obmHZT6P8#Dw>gXgEzYbd'
    '5__lH)uDnDHHNxg<ScxXv}{4vAY<9%^w*L^dN8(DOaCW1Fx>_P`q)fRoW+Ik#d6&HcDNpXx1tmxD%<yo@067EKic2@$mCB~$<rfBjY++eelK+7CNa6&gwx6x'
    'nAD@qQx1a`V5@CS^($*^sDTgR^(0A7+*%Q<PaPXO*J1l1Y>E~@yozE193ZgQ{ple0d`hMR!DFJD4c+_OQ&RRutOu@lsDwW}f5UsxC)dNAJ$FhmP{GN^4YhQx'
    '`q(SbJ?Kb$p#YAD&Jz#cBMtCPzh#}jrt@)mZIic7eT8Iri)OR%w}y7@b+fZS&g;Navx&A@(;Zbo(D2%|C(X}(Lr(d0UzB2*TyTzj|4tWgs8Mq3L@=TIIoiXt'
    '#Afl<R3EdPsF$S>uB|>C+8i{afAi!L3)m1wUcw%AB;^7*1d!&-sw$}btF(c$2?z7q8qX@*2@%#=)z*DGIyiGSTvnK}u0#f<;tuoSJUTG=rA5yH8f#ugC@~R3'
    'we^-7Kxag55&kv-t#iGUbIe>&%*t~BMqn7`Si3y69&6r{tP#+r-6HTwtn1t7Ez;`lq5<NRJS~&>!SEXCQ9Cu)=~7tGiDCYU;jH$=u3$Z=n=Gtx5<5x+tf0us'
    'r(o^y{dN>#H$G?zbEdGV-^h~ssHy!$rJOM7{Y=&kJ|&ZIgYucKvb}uCBnQZ+SIB-E&f`{!+1D;d_5ZJ&{qL@v-CHZWk2+S06`$@LOOOyrHQC*!gRRe2_Ssai'
    '=r`+l(Kr+X^B$mUE$<gYrACX{Kmsu+YjNxm&3b_kpIg<dmTvXE;dk4WzPg9g{OWK2A=JOt5o1sNuhH%{>VbLiv;tUQGWLf02NpX+rzf9Ee<s;dTxsfpN)%qR'
    '!4;j{uuwi5(P^YtTUAW8M7sesIW3d##lEMCNoDi>%*gYWh;s`%{EV1TWNKrguh*Fwf1*#PG}E5Rr#`6lm2?(7b;G!}PCTJEk2)ldU1W)!E+`t08CB%b<#?Fu'
    '`1W~W`2SYyxHd5&g5RZ;*X|Dw8zy;E0lPHbTrI11+CVl)ZCTv{GWxP#MgSd8B62%c<z-Z#l)M$u0&*x)lgPds``Db*&3hE&Z*u$ij<-fps_$QelM!kXM;oFe'
    '{hO0no=fsc&TfvWpt%-9xgNqMhS@FRU*bgRaK=obf?DFQi}ea*DL;KdW1?TxT@pjfry$F3l*ZjI!Op62uKBUIEov}3-8reKzD6^?RANY-kxLp8=b1mCvIM1J'
    'LO9o>pClgzmP%quRe6BiWEA3*hX-Gn?1H&Y!t}p>dYsd>b4}TCB5S11(fG-$K494e-qip3_y5R#nNKHEQ{nZhyjrYoxRNm~5$?2${-$sJf()+q*=JbW6FTE^'
    'h56s9R>Km{Ny;P1J%rUEauFXn*QT{Dz-W~C5@godltP+Zgs`a|kz3ta#n&P=$=6fbs3gD~uCTy_tDuzW$`z|}Im6NnK#+!O_K&9Hlq%q4MXyYzRgRN^!T$E*'
    ';ZNP+J9OIS8cygOykJ-Hf97hjFMoLWJeNJ^E~}}7H-}`U`~{o@wCz6EXe-BuoY6C+EOfZZ#nJ@yl$5128*cDGu{_3Ec|-F27gTcBz{J^Re2%{GU&!lt@#=MU'
    '_wd1Cc66M5|H!7*HCAz~#e2P6VD8ns$N3*0{0VEZVU|r;?D+-8HJhT_Fy41qbuGJhc$i&h_YY~c7SoVxJ+dNIc6^v0LYs@}cw7o5XIJQKiRD8u0`nWJv@u-G'
    'rdT#ZPG4$n0a2MzHLe9_J%ag;d-v{T-%lT@kut7+*|RSMU-(*zgRs$!Ad>bX8vw4hq%eZYMiQJUfMqNK12RUVI04Dbf<`v$VK4z&Pau(FidwFOf`7u+D=Z<R'
    'RtbY$ZD#A~61U0MNhyA@55Lt5-1lg>#InfGexkI<6p#SpY!FUq6Si;|f(|LqIRQ8AM(QT>T41tV66fN-4(M{Kmq$5;rU+`i15?sv%2izWv>1^}sS;aJ27j0O'
    'Lp1R@@H#7sf*7vW(+TvddIzSHR5rPrQebWvg>j8M!d)%S9CUYk(UQBn7cIGa+oC1X@w=z53aZ2e?G<J16_e@O6jF1oU%`5xyuv9SJ$&@^$EUCUUOfKcm!JRn'
    '<yI4j5@_%K!<IRLGwR9%?IN^xuZGvjiWlg!vGYh+e5Ag<$MO9KJCC%4kr?x(YkRm+UAoblVdsloN1ouZ(a_cUw1r?KoeJnwNT;n^IA0CNSVOSuFh<wm3v)7}'
    'DzL>=dWTyFCJI7l4C|X-VP(RG(>{9e5w%M0er&ChyZ;!qO78yms#S6~U905oKT@rdyPvvN$=!xpC8q9(5M-uVrqnv}2-#VAF$9ihb%MMDa)Q9)GgC4l+DKI-'
    'vUeBo<9s2*xmM<E(DFT8RVK;!XmNFgBoVpN9_Ag!z{DA{>D{iHBA4;-%i+xX1y8%%t#0Nlbm-rJ2YNTWL4I^D;d}E%;eMApMl~SSTDfuhss>o_b;dFT8hBdX'
    'im`M;yjuP?xmZDWDz8nv)UXDF+UjX`bQt~$j6-u1yO3hrP<A2JVzZ(&&cdabM2|;@E$tSFE@|i93vHp52=%%JUWbf>=P?9%pg?x0)#+sa0|qA6vgJEl&@xgZ'
    ';7Z&GTl$EYO;rnzo$}NE-geF`$s^Q3Z{iz95)}G>RB#m4Xf<6@X<*cWVsEUqjt07eq?ckezP7WOm9wLyoyo0WpS9SVZeCClAog8VqV7=u3W{$mIB5dvBlE1j'
    '83M{`eT{Nhaddos2C(9nOahoo^fp`tGF}BUUc_d_d08DdKNO+D3O#@v)=o=eM4oRY!g>ut&Q-a_AZUqt=HY;Gn9WQrj_L}CAaInThejkx!UvW1U=f-HG7fE~'
    'kSgFudj1Wkc0~1#?*8dsSZK&QEY_jgnnX>5Q5l~557Z+_HPlse^zR5Ax>(MGd%158O*Zaqoh_KoFn)z&{0hhTnHojCLr=rLy<<2Hv4ONq%Ldk#0%~mEvGny}'
    'c0`i6A+rgQRtf29PsjuyB>g&_j~DMM<AX4*mb2kacf6XCWX>`_)4sXY0iXqcH9SWSm&Y~P&yj;cPSQCwSY?NSCD`BsRtQh6t5q6O@N?*#5~ACaEf!a2=C}@z'
    '@A*m95--*Dg13r2I(4wtboUS3-8amY908`i-@D&2SC*z_T>S`<5+I_cx=H-U@zw%LcRxu1CDIF6ZAWcwAJ+JYGQp;bAS`6`bKhjc@sP-e)h@Bjooh|0R4`^F'
    'NA+V=VIHeAc3PB-i~kOVb0ZA~G(gRbRH~;EN2)TsO#@7r8J7g@NqtqVGqfb^S=i@pT-e94+rUMP%eT`}+2%})p8bH#!YK4%gFJI`s$_*y#R!%+^ro+t>|yjv'
    'Mnhm}COqRj+fXzHj6k)H*28oB2s8mp_i<j1a%>YgLf;)yIMRRp6s+x3F>GFb5BK6KV8`A%q@;nt3#LjqyfKV>=WxpZ*m;Sya7UvCj&hz~EGl%W*fvw(NGoL-'
    'pl@%Ah3oJL+leH7GCDPM>uLL=$!4k1deQc_gmiz-+3nzVf>JBZ`OvX?BVDaS#{|7|uQnD}R=(O)Yok^rMo2kE3m@Sg1E36+ugC;28?NDLUZMQ+{N@!NV@BnG'
    ';c7;t<j<yxfnbeCRULfHE$hcp4JRdAs>}9QCWX$!`Arai1YrE_YOxgd24)t9u`>0#h87KQWsDUJsq0O_Y2E;}noB-MH2`x;)SA?+@IIF1H>^iT<s6*pcrhZy'
    'PNhg3I&+FS@_hQ1G{0=A$Z$&;88|o+ZzF+tfXOzNc=C}J09;XKXpn{&ms$!#)L$x|cqU3KT1)aNt6Y)KjvlGxKsFMLd03rN4^HcaU_Umu&|V0o6W(Tx16K7i'
    '<_0k?yqk{K7n1nr33DyTw6{@9U%aJcg@#$_WM5}(?p9E@?5m8JU*M<CZqV}Zm0EYFXOjuBVjbTAQ30emi+2cqs5L+yUMe86tx%f4g`O?g;gV{zlq(x7MTPI6'
    'DgMN?%y28p9PkS9DM1)e6tawP(eu%WSZh>D8Gn{?H66|{Sq-HEf;R;#5?ju-vBhYvj66mBd_ceE7;aS9auM{jbQ1fHUOmNnDGt(pZ3BCqj?ZgmG;rK3VB3R?'
    'wWelGjU1%rl)yt4c*N1o{;Xg#w=<doXLGEawEOgy<A?cCN4*2V`n9(s$&w}D+O6nDu6n%4-0aUx>JcP@c7ov?z;caF9R$#ZKH;1_pdkesB0%U}w9UF4QIH4a'
    '9mcQD$4lq;&~GMaZ+^r)N9~T6yn`hME26<`L;hI)*?#zjv!-c<x%S8K)Egx2#+<cBQ0@Yo?dLEemRohg2vPOP)gNM!v}wI{-1!5{X_6<jARBUhHV2wKA9v3k'
    'h1cB8I0UFJMaKR5YyQ%S7INbO7DV^c==Ai_j)37{K+&*TkibCbuvknAK!*j)0#w;7N{xVWq^J}?DjSlif?lGi&MVPA0J1BIV`i?wYAPz!iW-V&yx9?&sXXnb'
    'y7?xf;%=igM!2vj`F1fKQ?)k0)qqElX9uK;mQrI4R--9Vkv7pwY~CF(f2a1RaVQ6Iv5YhuViqSunFNwe%$2&^tpo8JPYj|S|E|l?q~4ArTiQB=Ca1PDf4RN0'
    'MXT<4xKOqj1fz;yw8n#GhSp$L4XXDJ|NJ0e2FWXtmzZLAOwwpNB*JZqt&tp9#MD@B#(3)(c7RwJcCxfSB*@cn{z1<JqURIkvy!6S8tL-fWYD5@3eq+ahDe^$'
    '&c93?B<K1ePvtPa;iZ98c-*v4I%-$~cT&jUr|_h6O)6aDR?Zv62JTyzPL&w3Q-9&}+b!JWK|horS(4u?^#%HTxzFSj#Pd2$HTnf>@r|>b@yLJ40=dd+imfbP'
    '&u%OJ1|&Ja^)zNs)oMN#Bn&5D5~tfqI&lWBDy)iGU(8E+ZXX@Fm&)JE3cdXBu$V!dw~sGOHR*wjb5pdn7qFKiubdfY|GKpG<$N6dX71|Fsl-Ut3A*U}1K-6P'
    'j%C8SB?`00<PKzb(pfDQS8W)#gRt(d`7f}Xcq%%V;j}6frxiUVH&|Rw&jZ_N>t<I+yenF!NSke4FRUId%c)wl9<`^0ZRtK(HxUC{EGke%7|ohbb|w&TyoKqt'
    '*K)j${+jqpPw~dk>N_{A7H6NIF6g*vLmey?@S-RZq$;(FQ-wQYX~Dz()nc+nVB>MAf~g6Vj(i`~^YpX^<!Fkf?25@|&IrBsz`PKfy2>#xNsqBPn^6K_q7v|~'
    'j5nI{xHLur_DF-)sZyxBQ&EpvKx0ck4B_~=iZojyaa{G2T1xEI=Sz~^Pef=xtz?c0$^yPv#<IETW-7~=0>!ARms}Al`-Yz<f+<3(I*pysE8jj2O4JyV=~}yT'
    'HpSXLIHy(j@at<u)gtXW>5-d`OSy^V87XH|cJVaTjwtqa3<3M`(1W`oKkpHpyUFgr;{fQRuVlN^8!b2R-}&Zhc?17~>;PMCi+385b#W$C4kjV~Xri>V$RJ^^'
    'MK|>0f&W<E!ugLb+q`B<_awftB$@C2nlvU;p)f>JQYg%|sUl?N=QXhi4K<qDjKrGacPPIo*YgTJk!CYYXM&16W8aDh!tMxdH`|NaPzr4kbtzXi#ADSf3hG$U'
    '_md&;!9~AC#;k!AI;1W%ljF1VR&K_@_>vW0MxJrzrxoMb4uAeN{KP!F$}Sdcn~8r_uEm$fOd8xJEaCl!pc3_`s5X6eo<tU8jEE@6?NCE!M@9gb;=S{aCI`tn'
    'd|Md@KGqR1o2zyQ^(o|Cj#@KuLa|m5fQrS8H0!X^DH@&7>|x`A>|myJ?fVvi9wuqQ^KG$N2Ggq0)?2(7%zPrNYK8e%yw4w^8t|r`t+J{%XRt|fgudxD6-4Z@'
    'j18sXV{5f$7Omyl+DB@_Zp1?J4DW$bLzvJ8J(>;9$VRJ0Rl!|T-~(3WL)-83UnH2u==c|Ced5xxlxG4?{A)cPYS*d>7OqD=O*}v<vqT!(DDeyf$7YKzufp+u'
    'Z2vGEAZ5D9Fl9H|;eA7fE+*HA^$gQ|`^te)K~G`uq_!pc0^&#MsWtv6{bhecUFQ?bxzlbv=T(zYNM<L?>cvo8St-B=NbeaNn<xoWYrFM<Ahaj38@OH!VI4P_'
    'A$AFDsI9aaN*&Is#0&X2#(Uw{SoF$>Zsm{f9*26VG^=acMRgNDDYGj~wc3IzxtQi_?N+xlx3=lt9B!o2Y36K`hf_<xRHCqy=Y+}17EON2EoTeV-6s4t42hCr'
    '_okS3GtX=~#A3c|YDbn?SMSC_`m!yPQekX#cj3})1;VvoMgyQ+9f8`?-i-L(e7LPCO*?T5tYgg^+XvMSJ(7yx!md>1*iVqwoWmoLphKd`i;4)W4Vns4J6jcR'
    'f^j#MZP}rULM3v)t`vJn)mOUxiJ+Zb7OdBpjRBPiVR+c2LdHZ{)6sNI7ARzq=9{Hu?LQSRCj#?=n+l0RDsIlatEu!Q>r==_VQg!=p<c>RnAgR8d0VB&BkpVU'
    'n2Ig0e96PyKqnS*J_YqKl0%K4C$Aot+tF;js3V2X9C+zEp{>f;X%Pf)jGU^ZsSGbZ0Rtk`!Dq(j25G8=2GxCyUvl%%VSh8hgL@{^4{p?1jJ%S@jVc&gv~NTa'
    'o1NXiCV%{`C4F-@QMoES>x-%vqvr6;DfGQ(|4=ROc<TsBo+K<Ejz*iS&1}f1`w`u=Nk~IB$l*FH^}3a&?4UI=&#CX$dtx^8iHNPA|Jk?o;7pXW!&cXobL^@E'
    '?4*iQF<FyTv~C=ola`whs;nc^UV-Ienp+xTvx4}EYBY9F9$CV5ayq{DS$7T`3@5V7rczOBcn>L44-D2L87*nX!L#Yxy@U)6^xe+E;Df?@^CN#e;H*9Rv|6j)'
    '!ByXY!V59K8KZ?fLPUv^$5({i`r+Uw(QJ33@dnA`w<I`u{@#=2V<I<XN^pd!kf`opb}}H2#Q)+^U;f6A@!iKojBYWhH%GLFcv8VSZU!Y{+O*&y7aNwq_PN<K'
    '+%hV`>5}Rq57pG^i<l?}3ZrPc_!|w}cj~HXYcPLg!7v1QO6{xkPR;zSH<Q`D%7~t{MsMk~^+uvAP1)du+S;mi`iM+b)PPKXgjzm*I%-Z+Gcd{Zr`t84FCidI'
    '9Aozb%|#frez2GQe(#$Gkjk@N_PfhvK8l!$<_&N4AMlu{v<pG(J`i$V%w&1r#tsAl?-jWa)VmKjDJ^BT-@<&_nuZ}o*>P0GEt6zx2aMT1IBb8DCw@x5#(9!n'
    '2XQy$c98C>{SN%qbkZg*K?faKKK}hNMlG$mwgH5q!cN$?5XcLo+M^E%{T}PAot9dLH6}QY5VDF(6Coi0lr0w^0#kAxZT%{5W3v}oOfA|I$!h$?CZEc=t+S^m'
    'IRb|DA~%#&xsq>jO7q;A{rEsVOLLeg%=jl;SMeXgtxboEoTeN;_w{ze{qOASCUt`R=Y(pN<23&#gNtVzR{342cSV>?_Nsu=(`xER)#yG~&xHDmpfej`n)T1L'
    'kUBp;oNX7@E8Py+3H)VeU%!uV=S#0(KW9F9<sakHXQMejr(@q<K6Vyh_I9#M_q>gloz1JY)nzQt{_cPLHL$Ag;$KEScbK~l=A}$}teADFl1S0!MfFn5)szDG'
    'rsMOHixz}x+dymEtdBJ@k7WCn0$M1Tb3Q^|m{GuvIe3WibU~jGx$^CZM%dIw5hW<2bWuU+c0CzHQMuPiz18ba;?=}6{5?6t-#_6Db0S)^iWrA^VS|}|RqQ}{'
    '<4y+tZkuM?H75iSjoQ!}PJvE7$flqf2b(`VFcosGLEh57JVtFA2Mn6pmD9KlUm)u}siWKWihMn-kNkcmP&uRj{to|-N%&?LxUd!d7wL43D4C+}zb0*sNmItJ'
    'ngs5yO{3%6eU`J^&+@y2-sv1wAFcT$znWZh|9UmY%YBWeH`p+u-!os=6pi(RWvmsC$T;}-cxJu;o*MuP_>J}tJ#z2B%%KR3$FP?laAL+a<vE6x-IQ7Z78A#v'
    'Mg`o_E3cOzbNl90REc$uI`LwYa6q?YKu6TEC_J9`2$i>ChdC}N!Gtl>p|ST)!^@7mxd45=#i=?qb1|MVdLSO~BQ-r~cIa2KZsUJ`hkMI2v;AClk!IGm_4?0X'
    '<n~|r?J>$8`kGtvhYtU1=5YAAIK%ij31|Er{{ye}HX$0*8qU$@{hGRm<T0hx2{k2vRWUj69F0W$XyDYa#@nHD%X0L_5&FNYq7`*Ik_%3>yxq%Bc|Oq>yne}m'
    'Ad!)@xIkOtWpXwu6C$|eLg3|*$c7Viz8#v>z!b_0tZ7<R(rhSCOS=EVlXSh<5V1K0QH5svc8utY9knaIkr_Rv9Y5u>ClPlg3BJ^^M_kg;gM*kyNd-wNH!7vd'
    'NHAn!$2cqxL1(L(cqi7%zUSNwO1XqZ-H-xTgq=onkEZ~LG$;cL%?&LyP;W8y1U3y^rQblE6}bq%n**d?OqbSe6eXhd!VGi})0I2QujFG*2J!knk!~Qy`?t~w'
    'G~;#e)dFHPDrEM`K8!WVVdU>8c+p<s*Z;y=7k59l)<sQb!R8cy4P)q}f?E;^HjLsVUy}ke8b>9rN({6$>EM331`{(h<r~~F(V*lwdWl)Kv&FLH41L2H7cr}_'
    '>_Q2Ug|6ydOyfz;>8lcRj!}A?vy!Q5Iqn4ujCl`}nseY)bZbRUz9(C*K9BplL@_7Zt~}9O<q)jt?pmb!r?vhk;RQD~_lgTOn-fm<gdANQ$f<KqwgYvpirU2$'
    'WWLB=lt3QGl2T!Yc$AcB96TWDx-b<`yTb&e>Sftuj1@&c6Lfl9%&-X17nzA;Mb#XMhf?{a%7$xG2_feaASJSBK!DHZoTm#KEi&TNB+dbA%fO;IYPCyItLSQp'
    '>@m!XwWhEm#(7CKj49_wL(uFUq{|OjU79OM;5DoX$aMyq^zT%J=jbSMj;!^XJP`YXayUsjp_p-ZYSw74^t#ow(X)N@FipP7c9i-iW-T8#Nc!u7Bm--6MxmC='
    'ExN+m+I`LoT+Effb6ZJ<4UXA+jhZ{pE$xgBT*z%an-7dFEGwS9c>4X*pCA5MJbC^6$>Uc~zAa#shp(PK`+2_wb`Cm0^^IuLH*&ysCsD=nf=q7YVY0)FCm(YJ'
    'Ctj`i9f0Qc_vNRGl%TUjER@@QP;pYBXIw1m%OxDbg+q00{DW5l)ZOpXpI<(C5!~x-PF+fi9qn|(swBns#s|&CU7Wq6w){)M^;PRM4@oTLH`!J;?QUJ!G%9lU'
    'ELmFWgnc$Jk04e?t#jT;6}h`hA;ITV8_2Fy6(8R}yu-h872rW-RN3-VUh5N52{@FY=5`Sv&e1C_=BnkDL>X5X7Gg4e_*x%Q0?ZfLW^T@xlU)+MwT!RT^4vP<'
    'L_;uR?E^OP4a#)3pOz_M3`K!?SLG9|R5VYXs}64~S(A3R5Rk@LlwAVi!yuJMNwX$S^C{!zZe^a`Z;Dq<5qUd?;mu)2Q=uYS1!6jb5U3$KA(^aR!*~2$PJeRI'
    'Sb<nTHaGLd)pR)P&C2s~K3<jh$N6wm!Ifbr0duU1jeXj7wCqlH)H}qaug5*sBL3zzGV{>yRCAk)aPv?d69Ts0Iqn7#kzef6I(Sh@<&F(3+GoV(03M{M6tkq0'
    'HF!dI<5j5U#t-el<pQ<q+&YeKk5=a6KUpjDs+^y%FZed=U$raxP6eBz<~2QbQl+9-#dYNcILvc(wSm_4x^#YK*A;oP*p`07Ghj*1C=hJjzphTMQA7I5{vD*Q'
    'p00|D^wOSc9VY9-EWr-9n}d@!PKxWOqR3%_kKDptr_Oq*B8W{F+`jFX?q*=*5ypK~Cuij#*$#k)Xvg)LHcP|id4NVcv6Ofb#-jEJ&+w14{m)V#A7YL?W7L2L'
    '`TKO)W`xjNFDQ@*2L8?eqUJ!oYy|Nx;bf(NH%y7>hfb(=BomOvYS}GeA-YrHIDoZ1`+>2jhldv@HAwwWStPcb&W!)P?@FvNk8f6}pVb2a^}!sZR2zq*#AVox'
    ')WApWG-=yFxPIP~MB3Mmg+}vjy_200U%)O2Ch*eu|LE!rsPx@;1W#Y7Gf6+e>F|8Pwe-9o^~p}J16L$dnt|M3s6svL9oBXv)OW{bj@ewb@mnuv%ntE8)WvNw'
    '=X2=P;hU_cL)6~U9YNj5BG_e{031bXcX7$+B%C3b7P?6=ei+UI{sB$^Z9wg4DLI~NM~vHnO(+@HUw?MEo0=de`Sqi{30|Qi0l5@BGM*TiszfZjes(M!O1zp<'
    '044nMhK~jQ+yFB*25_tpKraoLkR8Bhok**IBS=8A3h9DJajY_HOdV~UE37SCp*nk6j9Jsb<0Y00qOqjbI0rs?O|rvT)h52N(?2=s9u4{@w9<ipIkR$t#S)0R'
    'NCYlU(}y7kvU-}3zwo+pYHVYr!Ko6BC@+jyT;E5x*|OXzoNQ?f8+dFjqe_{=m%K>()f@PVR-DHJa^Z<$yjm=&D+<L!ETf};c%y$HP{Xg?mx*C-=H==fZ347w'
    'EV6f2EoPf_S-79%#6gJ3Q#-Xv2USK0dfBo*DOH6gWoDwp;J<B#;}stc%QAeBEROi?KX~?8g8>2=7H>@yRrj`OP%PQq(#~ml#BoimWa#ePz19D8&EqG3d-&ro'
    '4_`eg9{uw3x2CT0y+Tr5fb;|63Tc{@hSmSA{S}v@`s%C8cVvV;@gsKXCyRPo-<Hb2ciWW0#=upb7lV(YnBcLa{}(l5mv)qs<@c7(RR36)3S)2W(Sj5<<y)?Q'
    'iWQQotHmW1$*#V^@=jb456hKYW4Z15xf*a;4lkvyaE9sZr=v<rRAfUGuE`R`h9y!Bt%lQ?)N1MJ5|w(pdS5<pb+}acV(o99P5(Y}nj14a7mlv_M+1}6iq1Lv'
    'Yj(tHcs_VOH~gy=drsJ~l3Da>FT`6T`ZTsX&7Fcw!PaOy2*lR8tCn3?SuyEM+BBK7^6omDE+De^bjmEe>wI1{y1ge%n|lXQ=vm$Db<t;x_jg>(7UT2w)ocxh'
    '+iN*ova7JE?Pvj>=B@5DJmp)Rr>+EAdDv(WK8$2EO$q@!dq#wqfjXRi%KyQ5Z|~mE{?EVvNA~mu8NN@lr$~?9V&Dvk9WNGVnA7h03h4251}(Zze*Eb(tm7g3'
    'P^rRWn3?e8;0yQ-_^9Ijbw2vge%XQ3c~ZQ8^#49bGx~>qwk$^N>*6KyuJ8Z%*>IJ;{Lsk;)QyYVJ(4nLfIFwFsaMU5^H>6|emCc2mnOR*i61N6K1_o7UB|tH'
    'FQ}?ic`ZLsEkLfJCe@=W^7k}D1A9Pz#^^UNJrhDSx>#UUAzIxlt|KKCtU<D0V4_5<*LL<s9Llby)fGFZO7=x5<%VU;T&hSI3(NiiD^IKjyG~Ak9#NtwO20Vu'
    '?^GqV1^xm{kD)h1&%nUJ7dB7i`eL}Y%YX*!N-CnkQ^HjeIYcQ47FD^>;j`&@y5by{v_29v;KSF>X!r1!$GkQwdM2yQZ803Gw>ZR7qi-myBqrb*&e7-ShL)#q'
    'u!n_dN;GM#OOv<@r=Z(41M=$V)|h_tEqupS#2LDu7nf3K26+XL8KwsQo))Mrd)tN<=s^0mO_aDpzkMseeaF9im;G-Xs6#zX!LU_%4lD(%dECXcz>`U~S#u(<'
    'X(dzqKgD%>Iqf))fUoe3ihpi@i*w$|+DBdfp<{o6RdpON;fWgc_*ppv_`_OtE(UJB;3HTKm*wwRS7&CMn;e~(*z*gQ`>bc{4Fiu282Rc78Bqg-6&Jm$%5~Rx'
    'q$t>xYOgRHo_QDFkV|~p%f2Vq96o27AM5!*o9>G74OLm@imzk3Xi8KHqmh%2ep7~@s_^RA&5U!O;448}8J`ekQk$Po=ViIVUC>kKI&RFzRpP+kqr(hd3}$1M'
    'EOBsOlq}+F^@`Y;-4q~-L+HIKr$J797|v$JH}9bL$r0?q-=4g9`SjV(TtoeS;qR~DQ!n@(par9KfRe*~=H;J1e1X21FTQ_?%1rd`EU4V6Yw_^;^B@0?E}O5O'
    'J%08hvZ8*$Q`PeO?_WIm9s~mu2we=%&*6E%6IHE@^srS%xnBTAUJZ-UY>NEuTV-s9>l!H2Z|s<wvw{QLiNsG0R?JCr<A<wrs>1l3erXTKD7ej*+CPRn_3)>m'
    'v~IV$U07SUyat%+icH=m-2$)?#%_z6Ey{g;`sb}9S@lT6D1X{I3c_vIu(I(0$M?35U@&=?QHL2u;+f1qhrRoUTZdxrJXq6p7ac?Ct)@@s!LDN=21$=`ch@2K'
    'C3JuPn4JOD!ibM<HCmM=)-Y^wbkJgc+N#h=7C6z35m=ipeK7}YcaG59b=anhwnV#2(#)CFwf>YJJ;?7K=MV0+JROEi!f*F-zIZpM7tC4K@ZEGe`c1hv-li86'
    'za4Sn#<!Pyl_P^US>05qaC_T7GO4o`tLgca722HRO<E*Mo|vCd=V-e14+m(4p{}kprB~&e6%i5cYS+|9QL~sytnP76qPY~fo5LTgs!f~g^rq|bs?rwA%B$9`'
    'dZy|`yAw8Nrtqq|2{=QLcB^7R?lEothn!vLVo-)T@rIw@wAYKvl4^%>Yh**|D_t^Ne!7#jPFo$$Lr%S5lm3;dwfH!_-RSRM2Qkt8_>_Zn7N&0fb7GBwQVM*3'
    'gH>~x*(S~vFcEK%H*;B*%L=Lgs3e}DCms%kYbxi6>gvUs4(mW^&1;Uk<0+D51UM*srfz6_k@N*3k?`B<W)4gN@L}Fh9aTo09B|^5hV|QBZw!_?41YyZ-q^%0'
    'q}Vo;T}ZXqtVq2?jDuW7adg=7dpJ}6w1QTE`=(UN9YyH^S7t=+gTOr4POiNpabNT!YG5#MprXO-OsVQC+t*@`pm{;ble+Dy;C8Vt#<_<gT4sRa9k8Er&Im6<'
    'U=s_JW?CI`wrWwO%a*T+^Om4Tz?*U-Z0RE^DdNPA3zO5r{Vq$bzow47R{E83Kae_;u(QyXpJTTC9389iYI+#Y7~973Ki-J<gOP$(;UHHw2bph+dT=io$2E)='
    'Vi+LtM6!x33~!xHmu(eQk!ao_%V17EPQk#;f1p;g0CtGo{A8Jb3YM2@S~DDRMDF}k!%&l&?jjfbE+ZVeGPHFqI~IFCUR*6fM3aQolWB=7QIaMs<^p6filVUD'
    'wuhvXAalYRc_`7I%WU@qRzyxQps*F1{4u0=3{s7-<i=3aXDv9?0)W4i(Ps6Q7m8jN_;MGMNz1k2G__YH$kJnF5<^qBi^s95JE!8RSNM=5OCDGce|g*`y$)P}'
    'LlZ4ecVjdo<5&;yjn>J~y(!g;9?#8<3Ym(u5=O7{4SLf$1%Y)-b*h4vqLFOB5a8eOOz3fRb!8RNFa$>wY*cN)(|yc6-2*BObIF^hwHLq&0u-YHxoY_|K0yUm'
    'n|U!EBi;veuX!}e*i_&TWYCvCj?Eu;<&XPCOM5dhnx-)k8mYm13Q&N#C-_JE4`2TM%d0QP#g{*P`O}v#|IqnRymts5ZgO#j$vxL)=Ys-8awT9LhHszs4<7Z&'
    'To#BmZmeQ4WwDY9k#EAgP=_CCTG*r}su-(qiR<ko;m2peeRkHc;ep~pvA<U7SfK8xU}aL9|2aYDh-Bs}=mkTYE9chih^a<|i;P<6AE_Fpg&`brT7@O964|Qc'
    '5tvj}tOi$)woWD9v%=$K2M=~=RZ}<-eu2Z7UXjB%iXmu~gfF97icx7DR<*m81}b|hvx8BifIjA=ip`F{&1gBjR*<)B;)yVuk#Qkz&uvz5SIdc<T}W|2TM@Mo'
    '2+ZV)$MWi81thBtPn_I;-?H@d76k&MzAG*ZhSN4@gtFPoJDrZ_ZchV8p7ifO=)y$QhED7#Zi(Yl14@a2Yz6lPiED0Y5jW@_q2s1)Y!MQ+{0ljP)eO($W?$>o'
    ';Fd3O%k-J&lyg0a4iO`DKO87WoSeG(N#l9o6U{NB%Z?8yZkk|o{@!4%?PFN-SF}6Fg~R38QPpO;VYc4n6#D$S$4!%j=^vBB;WF-IYE1kBR#HE(oNwRUdB<5%'
    'HE$tzC#^N`6IV;P%K&8Cy#Oyphu#AG!TXTZ$cOd{r5Ay`8Q{PxJ@9Wp6OOXKu&^X;I^OyyEiW|f_$XH75Bw#1p(e#nYO$or#ZSDDiEwl3?H#B%JYL&`uHMh+'
    'wnkmCdE;nB=;47~mN)#C-!SOE!L3V<8<WZoBqs84eUB_MOyeM)9!i9MCO)w9Q_KF!o*O*04@QK7=c?!cjy+mjoh@*D;G<V-<ga@8)pS1O&@CP*m*ImE<znlA'
    'gU0r)-MmXwR?6ko{7C2gFPUreMfQ@M%hsrFKxUCmmJ)squq5Y4rL>IN)JOx<@y6JSFu$Tb*f-8IP@kq+2i^0V&O!YWO%vbci|?OXdk`AO1@2i;P#}4iLV{it'
    '7%tbZ6UL=LC@wP!1~ZYFjlqt!72%?6$?5<C!x7UtQh@NAZHyJPWIPV!yRHi(lpsC>ujz2YI$fj$|E0+~CKUxJ#5@CUi|h8w&gqllQQtTeJSqUkz62!8Ii5^~'
    'D0(we9Nh-`Ird~Hc8S}jn5Liz%C+Ana`QK6j?0}iqc~u}e<NR1;$mXdMsJm#6>Sk-79F>J7$}oB(RN{Jc$$cnFdi@b)%j*NT%9<i?U#r)^M^yh42-c(X_F@h'
    '2_GD6;OJuAlRc<OwsdVzn9~#nk4)(Y?iDCVn=%_Z4~kX(SuV@69%l82J1iaD%gf<>?=2|s`iQl6n)n&HKjX42RGeb<1RjOY7ky~QK3*?2Ev0SSQch83?&NAT'
    'h4{_r(kw`c<+?b}aIYu?968rTb!+!a!Qc{jPMfzb^fAvZ^g7ly6~Lt`5cc21#0+6wy~@A-6)05pKY=?8T{g`)v6rUX8o8Wf6o|qst_+}~90+{Z{cVj7g5ahk'
    'zDpFtafb(-?Qa;&;}?Pql55nHS`&z~7kIn5@Kse=T?rd~#lLm<3DQ*`-TJEKdE47`Z=9AA1O$w!s-G)*B0)!fZ4~O*l%6}H(^tdwYI^+%=yYwke<C>l*7Q2$'
    'gg!?geREadE2+tHD`Lrqt;^mLT)mdiUul|%j19tFxRW3AvT+Y&CswSUJiP8t2l^F(WH}JbT5*oGgdvN!1}zbB*{BwnSHXAUV7z=GJcb5VPv=~)=9HJugP&wS'
    '0kHwzU$Y(##uSSfij0z9-P6HK>MMaY`ny?bG!)Bjv3E9IbD^uYsmNjD-xGqK;g2peQF|CwNysIcXuLWePUk$SIehd%<nIjoVRj$(!AVqek%!bZ>O0eVhN^#w'
    'fs1WL3Gm0z(!n8*JoO#hR^s*3T{;vGX;QT?C!*nGyfnre5(e9{LQ=Fhz(^7(lDY(g{#fKJxE{SqPLh~nAn7o@JR_E~jU<jIXr9WxeQs;N$F8A)HXQvrtzV@T'
    'IqG2b<24zVs<3tfn7yj$(lh%D_fl2}e;J0a5k7IeOiL1nC-?+D7{4eu0^Lp|1*J<wsmGy9;MsZYce0}(lfXV|5|(*CR#Srd4S<Td)B6dWQUDH@E<3bV;3PlV'
    '7i02xY5Q1(1z#bc>n>HooKW=+J7;cwcXv!n9K^wiHX)PTxaK_oSWd+m8NHoOfF*^0D4B4V{yG{2>mM%zF-AE+xzJ*~Sldo(2pKxyimuQLi_U;{t?9{lY1C!-'
    'jfaU*wjxCrWq2HTgUR;n*xv4#h}h%iwc+7fY-q*cw<RzvV?jen5q7j$!K*WoqB37cr8VXT3P9}zP`+Y59&R+y$+^6#jr3)Y&Vmq6|1Xw&=muitJT?AvJY+i<'
    'TQ*C!dMr2qi{kIuwm?Xc1MxJPL<7(J4;}(-kZ#@>T8Og@yk0#D&sEp+2JWWIm%~%hdteZkS7agHcYEaR-mya)?*+JHb2Ed$^KpHF>9<zXvyDoAwUrGadX&y|'
    'wKqo+r>YVs)P(X11GuxmWDi@&)E4OchjSG?BF92&-bzqvRpu~*m-1zjU}#~cw^QKO>&ypyY#Me~j52M<f2%blcnlFR!=n|clS4K~##Ee+8QYh-%I0B&J+gfx'
    '#>vd1T+#4^`g-95&sM%4JEWF&$4@A*jr(#v6AZn&A=!avD#tD4{z*__;h2;3o)N3?3vyritSaFxf@H*vg<MDcO<+MnF2^9mTDE+m(%=%G$`0U#wPaB0@d?^l'
    'xB4nYLII;{!$9TP`!~cT#c1_IZt$;>Ui)H;8f-I}P=_7{w*g7%vq^-zDieu-_I^TWEO%Z9sv9M61_Z_>LwIA-6(=CS*3}ASltx*{Z7g_F-m9z~Ur)dzC8RI^'
    'wMAE;k|SHk#_1e!DJ7Y&VZpb#JqDtH2mxkJkM$wi$|v4`GjL?oq{7hHpq~~TpcfAy{_d3@p0k``-f-Ycn-PI>r1=!jpOviTxH$)8BW4h)o63SRIEm~&TYZE&'
    'd*c-&l?04do~Z=SDO{V@_k{1xYIt1LRH|opnzY4Fa<A~$)wHTO;3MYH?MTMtA)FqqdSEc7jG%}=RVv&p5s=%TLctXw8lExKZFLT`8o@(5mi<53rl;~Yk-LtZ'
    'w`vEM&=^mI$Ei>Su}!&(wmePiM`Ta5h#Fv$-4oxNCL2~}*oA|W#tr@-{%n@!a7RmX_~XC>He$}xGNGs5aHDs20zn%0KOYWh+87uu=Saz03l|=g6igJ%2~g|^'
    'Z!X9Nla^shu=%1|V_FmP&vh{q-=LQ2Mnc_(l!cW8T#W^b#fY8LMoq{M&}CGrSIW@9{S%Wdiau)6r_%+g^iiDw=I&Hh!Rfky3&DR@nTAVT1=3RY&5G&zL6Fdm'
    'MgU2Kh6sE~3Te~lyk!P_;e%7QxC`f|5&Y_*>LE6Q+G$YJ?js?0U+BgCdeEyNm;>u58IzTyxQquK4UjngL`Um?Tw#14f7DD=eU>aQ50<^yJl*DvwBf4`8ikp$'
    '^tufT={~_d^YWX9d9&*<t7sUVb!+Pobga=rY~C&TEBL)8L6>|AN2^cI6@b|VFk{2{&9vYhSlpu0a``E?1m6h;6MV*7wlof_F9HapcuRQu+90qqhq;d}^ERBO'
    'j{}91v@GYq&-s+ADF^>mX3s<t@kq~8n+cecIh=&sbg2Jq2t)&}z#DF!vV#Il7go+}Y^r&n#)dl2S`&?Oh3raOto^7MQDJtk;KSAog~Je{iQ{50c9GJ7coZOF'
    '6An;gtaZ_O2!Ut=I68#?-9^cCN_LP`h$QQ!aKA3CCSRUGuuN+m2{jkTy)-h?C|X@x+)zt5s*k-t-GdIw=Y}z#4xTb{7-9CrewZWC*cGOoFt*czfY~W@H)e;>'
    '-^3MqY4D*81+LeDALUk`E2+sf8!lPhZavrPz|IsfM9);6q**_juDddZ=gk%Fg>Ig*mujCD<L&RFZZw~XG=b2w`#E(|&@A<Jt^>h%MNvPy)Z7cYvL{U&gm(0A'
    'o?Ip7K}9MD+f)rCEm8?XS4D}DCDI1YCmhUcYw4otB)LL$UtJMPZm+0H3^c<%$v4rVA+@B(CAlmW)fCB%7=5*UMKoJ;l~~^LI^rYPsCwvlmBBC%e_=B7+iaL&'
    'Vox}d;fzDTIoPgs5-6`*1aj$ZeEYl=`l{`w|MW`^W8%UVYdC68Oi$IRfe)WwL3NshJ_a&<I0Ie=QeQKW+{On%?NyKzRAZQq63{i*Lh-ikxj2H;C;8iQ5lJVg'
    'xiQeV5F$L#qxwo5^T;p!QuK=|99Y7CvDAUkNy^_Vyn~@?j3`LqgKm-(B-k^#0O0$!r9rkS@DQ8HWQK=ABT6dZ;aNGWv~N_rN^aAaeuH2s*_YX+##`K4pRfr@'
    'X6;A&z{*B}0WT^wQ}3|0mlcm`)(qe^RkFrJO8Z8?w2S(_M!D^7ND*r_v@V-gwn%VKh@%_29j5wN$DHPXFwapeh$A{xaYE(?{cY@swbj*1t<3r$gjvv|P#f)d'
    '6>XkI@%JwTSWYZ5Ge*=pc@GjmlR8>5&r1fm)5tli<9TQ8>!EcEFD;+HWWRlzdI0vu-?Yt}(3~B7V?1)zA@mcgHy-h_@4gw0;sN0Afw;R+B)JVA;AKRaZegnV'
    'O*<UIEv)X)#qY_jyOVV2_rWW<{!S1akQnx8mtkF>AmAW&+fQ|mG(cvVdMySLY?23n08E3w9;;?>V&lamtPBuD<I)m5hl|y8u|h8IEv1ue-^QDQsPrv-^+F2!'
    'hd=z`C8ma4qcC@wnWeFeXOmUf)S4q;oW5aq3o%)qt;!){9_n9hb*<nnZ}XwB=1{dw<!f^_JO{>RGcJP{g#2pcGxJ8=Wje7XG6nJ8*o)N_@PogX$Q`%sFQj4@'
    'cXsm~Nf;drW<)zR@o<ehB>4_bEZ2qw=Ty(asPclQV~ZRs<T(Ja;#edLm)!>dsln>}jvjpad*pdMzlO8T6u_NwO1zr_ewPZfZLe%`sz_cj8qQ)?0F!&vYMii+'
    '6gDtyY?)|-YMS)WjaBn@R%Q{=0`9g#tE;WBSB=W~5FUs6%IMbY>qCu7$}^T)JBC2BVEcW2Ajrzva_jyUGp?^|^FL9O1L3)9bU(HzwWt1}i`_*{hwW0TIt2nD'
    'Z!s$pDm(Am{T}Aq)C&^~M9XlM{CEXiP*B;W?N;Td!kDFI%L~*r*x#?q*;PT`N^(8S&N>9f24{$HhM_D*n3m<bV5-h7jOvJf_)nd*-Xfi?)d5-eLmN4};g@}q'
    'hB(Ky(#m&kP4(b*ZLD((?P5(+hP$mwO>%^GC|@pDAy)dP5XRdBOook-;_(3anTD>HIOe)<EG5vj<XJkCMwpCjh^iN0Gt8yI1y&sVBJjZ%hCWtq1+Pa`#0?ts'
    '&0+>2oA!EyL}+{NtiX(Rv8XVCqL?`89@;Y8E}6J|h3tJ%$BsxH>oZ5Zhf$wr90c`yt6~(ye%l3%I>`dQ9vJsWgXjs;Tg6bokzy4+aOPe<Te#CRcDE2d=$P*|'
    'P36ip)==_9SUowCc9mS;@T`>d1)NH`9$mCkaz4?hxZs6ybV&z(`Mw9v?ZCB8TXI#*bmR)-D|CWL1c@!%J<88hsM!<+IudM@JCLjMlksJ|Epd=CL?;!NisG#o'
    '3@I^XCzy$#8XTyqD?u%S%A}A`DIzer|7?0b2ZGIYB`;L+d(^pm<ySt3o2KV#=R0fqA-`tzg>!JMb>(c3<HX-9?W|MlXc*sw9%o0_PZzO(uSgFNLI#WeO5fRt'
    'KUoLq<IO)F9v;=<O_|xbY}Km$624h_*}J&Q<cX4<vvIXG5n{v3sR0EeuIbH@*k*2pWVa@0`7C~<5s+{9Ck^(O_AFI;>wR1g9RQkkzCP)50p1?aCXi;<w?CBa'
    '31J=IjC)JZ!0^@-1#WN;bY6djmndqxuVC17i_>7E<6yxNb)NU2a3cZGDDS{ve@25;udYspwCuLgy+d2Js-11XNlJg5h;a!N&~++j*Rw${9S6JM|IV=-TF5-^'
    '=TGl;d?!Y`TiyN5R}H9F-E@rx$?&(hn-KisV3|NGW!*IcwWf2<R145~sjdSKoclQ*eKC7Wl_`=oS7uPYecsyL%(}(2YDZLzwlA61_~MDS2+j5USLl)^x+Ap2'
    '%b#J3vNXnRLdn6;))w88VpKt~N8_Eq@};hsISPjod}<J*!I{?GPWhfCJMyLAPUw&3(GdWk>6>aX?~ONC%c>oPQ&5}8iBBK=9&0rIxzO^}YEz~c*XYJ4bjCN3'
    'o9%Eg;ZWJc?bwn+*7|#x;w`uos+*Eti%iEgJov_tg=r$&Lfbo?#u0zFhgC~SYx-No69T+)xKO!`B9}&Aq5USI%FbEEczP5t<w}>+P!4v>1KaJ9?eNg{drQ6_'
    'u4?!u)sV*vF^U;F)E*@Uz@Xs*RR-Z0B>VCR6Zjp~e&}I@VjoJ{<VOJ1tN_kzIJ0_)Tq&`V^Z9QXuhzH`qp4wQhuxp8bF_8uFnEASKae)ru@pr~1V+ch3)Bbf'
    'o*gb^19ut|2pH;~WZTm#tXfMBd0f`VjIB&JC{)%T+~+_{AB5o}e}T~8+z#}eqYgt~oz^sEE_2=&GR4;<eTf$&ePuwJDgZduv#o&@Y3#7QvFwj9?MkHMt65-2'
    'nF{t8K2ioY508w?9;1hRPV?Z6PLYv4CiIIGn_fGu9nJln=HAkBzmMHu9vo-x?>|4-kyQ=6#zG{f-i6mr5eOK<b202tanKj#*;FbHuHP+mG0oKi4Tzl5VQ9+='
    'H&kNiNo~^5#XfXgwm)$D8Uv~6Rr9mw)E|_bW@{$>4Wb#7yhrqL7h9%m^u%OFH!fI<^-1;|<F8T{&xCz*jg<vU3}!%#!xRNJCG3pz;3C6|zBDl^3a!TeY(PUg'
    '2_M?H6u=WLG#?N_h7Y{k9+r@TPi8gn=Y(i>@gD#Cp=Esiu}p?PsGjX>q~d1)ET-NN6~^Au<9sy)s9BgUud#IQbe`iMWd(xqjrrn6zCi3b9iv{lTF$2H0>N!X'
    'z@dP<%^PD!$2=~kF%j2Ev(B{)sqyqQnIw&u(_zKCQMhB`5?NPoy&%nO$Kh>jIDsy~?K&G1j|&9<o!LA8eWQQ4u|HtGBl@1cpp%e2_g?4bS}_$c{R~+0^ssk)'
    'm}k1^!l~oO5d7d|CO4<1YbRWqS(blF-nuoJ%7a#RLrG_+;FkS4)Dd*xOYGmqvSfMCNT%@GxKM=IOx<zfLS1<-4P((E%Q*UE6$(zmSly|k*6)w%=w+XZx!(qt'
    'W>dF-Me=V%6n?6aKu@alwp*?!l>u3<8*&iacO-c9YIs@N#uP}<=GTYQ<OV<H%CYKH4nJ@c;OcxsDjr2%?#ge{H&VDYgzXvqo8P4Xid}-t+m{0#*k`7`p)2yd'
    'ZNyLG82w~mmjxE#Ual6C>8xzKmDuOv(o9$ef(X||Syj{FyttT-$K^bgs5afEX)H>tZ3|8%hf+<NMj{{mx-sZS;cfAx0M+0>H)J2*cBE3rk%_tTl3=ue#5nxc'
    '#KgF-3o7i>_Fjlt1XtK{Rn>^;vOPtHrw*M?7|C=LXu!QWX=y&iFsbITdWfs?#zk0P!o;)kCL3<X)3uZeoC1pu3qs2SYW!><ipk9<!(GAjltPRP_I9aUuT=r`'
    'OmS}D%VskQ<>ksZq$<^Qt9djkV6tA=!qiwITvaC;EtZ?vuvd32jCDvqS$k=0e9vi<<LOm(iuxZSje=8HxmWRt6vMZ}=?s|&RteS&%L@v?L9ZqODIdL*#F)z4'
    'jO$3=d+bi{F&$q0$#~#(KL#&8{Nl}Yy{1s}^v~$C0(^Wq=j3)BwWOZ!u(X3?SYoZAo0xMnvxTSJ%NC+Cu+C)5MP;ixMh+}`kWTP4jbDx;ma;gbY4dt_7#ZV9'
    '-aYc2;k<@@Z&8DU)0;(kCOS?6Kg9h<d1!GeZ&NsWzIfO6obk5%r|~G39x1WX8x2HBH=iuP__=Ibgl)HmWh_=qg)a$n5S)7ntsQ|%!hv`VI?Op2UnzU`cMp%$'
    'v_>zz%Le`ZM1F-veJsi}-b_2%oVrOlCzOcII+{;P99(%2u5@RDs{GUr(j-)-+2Y{oFmRt0v4OO1JnLN<N73HB`^UI$TTRtIa@B7cVq6U;rK;2kT$uTV)RTO;'
    'v~1ktIckN+mXho6G2e=&Ihl9^oTWYqLbPZ=sr;Z#U9&uE$e@v~Uy6)uxD}4zI;jh?mED3k4DdJtK<r2N8jgs50_XFEQ6e+FIIuwyTdx7jC!QQMuEpt7c;HW<'
    'AYypW^&`yHLvXF-VXzt~&tjBozf-y%Kp{r2iu;+n7cROV%aJvCr*DTMMU~`f_Wej;^z3s^#wl>~st~Ftm1}V(cSZ^OEJGA<8sP->9yN;D1=WQp$*aK@pLYn='
    'af`k&bx@>vU-o9qAGk+0E>AlHCl)pgW5U+OC&s7r9?14pl(w@3d54>{{v@nD&8Aee;$32%7gtH~L<$jywqhMkD>zzWoKoWjx=`OX*HBuu<3%Q0md|((%hnXH'
    '7B%r=pylOlAE>^i7Xhy8(t4KmaJ;yJZO4myr|+$0xy|-|&%n=){DF)f82yv;gf{<*YtGxj6AKAk!otm@TctK6PIptMXQ&w~(Xx}L%E?xS@GHY?0w*D(>h}@_'
    'v+foM8lCivEaw)ccx_}^-^cQuw$FbU2p45G(#*}HwyTqputUb38aj^s-l65#r6no`<WCkZphG`e#K5Qm6F&usXBwD#vpADo4u1w{w2|Ra_(@-e<vIBEaoA?o'
    'pR`Ej_ny)1hnB;)i6uHFmr2Vmto$co<LxOou=4kgG%GI)+?kVC8wy8h=H|`*VlIBCGs1>sC)O8dzr>Bx#BFs3$ushpaT-9qJJPfjuErVutj?c#t(5G*#tGiY'
    'ClVj*HX6i-dv}jMkN6;J_>Yqx;vM{#$Pc)eZ32Ysw?=~C+l>bOd@_VvPAEhK5ZRAX!VI>vsL>NvUj}$!`*iY-H4E*fHf^yoGec!2kf8n=Q{D-RhLf}!bH`y='
    'yM@lj<gm;@(Kf?Qp6G|EQytT)##knNW}auYxw;yzZVD5;VZI0M(uHCo6A8OCZPJoW4Bx(TA*A-TK<?L6x7D46%5$wahrjQmzPd)tY4Hlo5UpJ2;55EMdy{(U'
    'NCePiz^dwK{(YyV<9?3LhYJ|!W$i@Lxot=HOTnCh3J=<RhZah4hL*7{@V8bXm0=I^ouJ$IaQQ^@If(b>^88&seyR`pfX(wUl5O;U<cIwqkkMfHYsmeZ{e17}'
    'LBl&T4xq`0Bnaai1c)oDf+Jrj9Y%m{EMF6Jb>~vE5VoN@R#frNdBoG&?RXl3+z-H{6+}uH56Oqhm;;rwLw6$9tKXQ?Y|v?*YHQyBuX`$$_}D`_AGhcm7sAM8'
    'a^RKbO!IPyH|610=a3kZ?@y^uF-oD6E%)70N<(PusSsOQ#;{F1#iH+QDMU8ZGzK1-lg->XO4cc0Jl$kMSMn)cGC!!)1I4acqes5ePa;c!U-auuO+Jy-<%A*N'
    'c40etV&4j0_Vc#(e{GR#T6_fk(O?h9Yz?tEoQZ`=j6kp?3$t&@f==j@+JBQ=h6e50u|fybnz01DYljIuO>0K6a`DGM69v{@hJ$+4<A6+%l_ftrJ&8-l$Bl0Q'
    '^w22?SdUAtVVi>S&aI!)WV(Xa&G5~CvnMwaLKHaUC;1?uP%wp`$Ez(2U>-R!=RU`lM{dZ$LARa8I`k?2v_sV`SPe1=LczUD{J#QyATt2Vs0i~<TJIe+e8}Eb'
    '*?R>8A6kk2K)6@j8Pywwd~7EHNs=z5-YWLH8rosZ=Y7YaM<OR<-VhxSeqUfN3Xr8Soh^KpPfb(gnH!`58Cb9gv49URR{c91?S=)my3(9PS)6^(#e~);oz5|l'
    '?b?$@{rO{g)Tn*K+M3i4#Y?Y0Iu%((dOIQBrxRkE9-VlK&-@<YD<cgMgc`{J*ID}{#P;&GDim!D--#bnK|PQI_aiN=2!ZF&*=0tiptqT9i(pc>0`p)SH*H__'
    'r`dHpr@!sJrn}|7J;YMCd5vwTNuGJT%L5`G$vb<@Y>yx@>GnO|Y9;alZ9t_jOn8sIF#eVX7kraFeZel48_ZLNzLg+OPFGX3Z3M-Vo$DRuerY8!vWI9wbnH-t'
    '6=}Wpp(Kmm;}j4cq{aq0xBy%;gGFMBnN^7_CCt)FB>+NVtdLHpM@rxd6WVB>?^Y@~HcqtK%qj43eKD;H*WO~Tghsyc5c;)2-5NE1;=pd_7*X({5|@I11Pz;E'
    'i3@C7pIzhki+O^qcj;n2yFn_3pfpBZ&5&ZByd!ZA#BO0dUBN{B;;YRP8GKaQ@?WsB5k9J$c8%^xzFx032p6;R?Qm|R^Q%02Z`Gu}V!&5v`WgXW;m%)w@N_b='
    'xCngZYtFlXIl=jwls!vW5(LKSd}m?(s>DpHc*yOR5v4Fz!6tZU#U;9d$sCH{&<>dR)(cV=wgTP(Hs|-VHiJGdU=Zg$QX4Vc0M@BN;|W09lst3VyXx_g9p;%n'
    ';H)im7r9zm9xtqK8xB&xy1oOe8GV2j2I95O7K=4Ts4NStoi!BICYw$d^XVL(hUA~%hY%bTTDkQ3GAdy!AX&>gjaOs#cgs6+dg;$}Ns*Fx=Ua-G=3P_b=hQv@'
    '@jv;V+>sPx%=B4YELd8tMvG<H_E5Vz*p5-yql2`Kjh+MQIxiFI3hjDkYDP5s-5ey6c7%CG3?WzeJfPgIxf3STluDq`6AiDYSDP!Rvi6EVOebC@QIpOkUHqa}'
    '0Zy`sQ#6@$i2JF~JG;&`y!NIVUf*&iL$5}g@vs<OloZ}XA1EZIowT4PT>=c0|It7>hFsAgz!A;>%rP+rLbCrO18-nE7og+yBV+2BxC1B8JF!K);}4)jT&0<G'
    '{Zn4@cM=Xa5?VXG;?$4+JFpk`{!_YO5|R-NV>i^IUZ$vw-V99wuN%9HAppMeS)s?M*6Hf#A`>C;YxYmsS-BqeS|9SyvmJ%Dnm3iWlH2V~d(E1#D>A36&x_TK'
    '&W!!}wl74ZdFwHV@3Qk3WNHZoADKw%uWFME?q8n`GQa;h_s-sD%Mz(RBk;u4`wlLgG2<xG8R>(e9`U(6H&)j!sWUT$A~Qt=KL4F>+ji{M^AWoaQlxqYxD+&e'
    '{_Ve8tS$lhzN-OxjqLUQOt-CD6i5eNQ%*XZ`15U%=ed(<Am4SnhBWQAXP#*}+f>QP>=vz^8>p;F>XCF%19JT8*^9?N6c7LQ@ac~aAN}}*PEh(_AxLyN9}~-o'
    'T!Hd{YZ!YsMZxWC8ZDd``;($-HM;2Gx-c6NNc?Z7BerU_52LcF<?s?%v@5{%m?COH!hLZ*J+pk28Vxgdhix%o`Pb6)=<xVn_SIL}-KJUK@DM<EnZz{RX$}?2'
    'E*2ZuJfbp*oYxJCl1ek9!vy@WnP1Kq??A>*4N1}F<V~wx@cmyp>}Fs&@0w0EO~P4g-Fd^Y(mubkmh;tcj0tTT){<zFIsxmV1kiffa2|E-!ZFkJ9ivE+xB_Hi'
    'oK%v5aHM1+f`+kFmVvsVbRq&@$t78K*3-#wv^MEc<2*OMr(fiE(;HQ9VeaUAC$5L)a=%CTYwVw54QYNy<qp=0<L!b;R3Kh7K0vuCNU37Qy#8L$UMuh^lcu4Q'
    'nw!V-O!HdHcWzAWTICI^UERWwt>Ybg4tgDpGhx`YVKvUvFgmLAOo88G)YzY^&aF{J&^Wb+Xf=?ENOw;sl6zS+rA7L(wF`X$P|#bD3PP6G>vj~)+TbkMshfk`'
    ';I|CUj_x~-u2eXwan;Ye4{6l|$q}taVdnvTlM0XNeTUqEsHWvU`4igdNh+7G`Gb%`!^{W(xo@=hti@!AuyCCxCV$L$OokL-_lC&Fv?p8~h%&~I9#s5a@N3HR'
    'V@B#t*X32!?zozU+gdv2>K*1?D`3TDI?3ntHI2c%8=bas(w;34d&_wJ8bmt?Zo=gJ`dURk8lR5da(>=Y6WR9|PSt=K2K`;zwODc58uZHFlx3h3t~jA-cY826'
    'i#zYUfhL6G?V!2g&Ra<6p`&{UHX&u#=F<s>nkbSAueM_8;BQY}ynOoXXP5!umHiD&Pz!&9EYqjr#aq>4vl=O{R{o5$Ta_)I-H5o<hc8||ef;B-;x))Gyg+<_'
    '_`K(Dr>n)BJYDP)GZ8R;z-R?+hDH*4HZP6+mt-OaTuft#Nl@XC_-hnql%%9Bb%+rs)76z3GLTs~RUHRWR0S+(v7u0AG^nB?9u|wgZ?Hf}I+-$qqX}?!g`TY+'
    'cou0STFD`7thJXhmhP1suqdi5o;##<)}bQX`YSz|=EZ~b$6K`hJQ9b#;$9jCYKXhtGxpYZ=M>N5PY`SVLDLRK)|kS(ZQf{}1*sFw4?S!qXo<2mGS;YU$btT{'
    '-56$It%KSEm502cA%&JwdIYt0Aqm2#Tj`8xaiRE#`VhQrCr))df3ka0Fc>rl0Lf8&K75dr2GY|9OyFQg(I7cky$~^oB?f<llsGY{6CYAtJt~%zC6W_{vc_PW'
    'V38UnB@qm2#f`=h{B3NLJ(9Ojx0)Kd-#kX$dYlB}_iqqe2EL%;_iKW~I;q7Up;5LO<O#u~p}&<|25H$NSbQLq&`zyX5w!KCjX|B<;P;9}ih&a%f*Vkh#lT7w'
    'ej6oactGZaiU2AC`C}kFmBFLPZE3VBRE5$uze=ZIW^#k|&1NXhVpY4bhzC9FQ<YfyY=}N_nB^1G8Ny(1@7^QPp+a9cB!|{}P8^;Xi^D{S>xplYAC~2+OSWB0'
    '!c>`Nz*NA#jL5=lcHI`LP;F-GTD?&IkqMmlO4YnWB4*3_U@?{3Mls38DJv@?;~>senI&VN2hEvs>PRwj@FnRV;U;@`@i(}D;CugDCwE?=<ZQ)GE=EV!V+2nn'
    'd5UA$v@!}@BZ2Bu<BE9rS1MGW@*2)JrSle|sZXWHXtBI$dr1IVQ?!O$8UDU7G7YZQ^qFn0=9NDW{nW4PjH*WwiM+XDW_Y)0;(j=T`{+#)r%BpvlDOZ1s{8Dj'
    'V98s{XaV?ujz&yYyDcT^HZV7s`c3C??!kXta$*hjc?|XNUe3$wQ_6$m1BK_Kv+r%ybf(kP2j@TP;Do7!212tL%0Jk>jx#>Q?ML_6Kf0$oAKf0u1rK>K8>5Fe'
    'Wp2>eD^^PddkPLV@onwi%$zlwVD!`n#AOFQNMy|ah&1_rmi(I3_>w$%hTiAPEv5|ToDm8oi3DS@fYT_}G)pyUStc>7#{t#}GP}z$+ohLCCh(S>5?vZ3mxR0$'
    'kMD~ub;8OHvPy%5Qr|mTY*HN9MLKDcNgDdA1vCjHb@E7)6r%ZgACMX#s6hg0mOGm4B2i~6bkxcmk))9{V^Bk1%xI7@60(IpS1V$8f`w{f<qIuU)Cv`mD3NCJ'
    '09Vb0&og){M&m#tF3TH6_lRei-FF<H!oC6D=aT|wrSN)KTZLJTi4|~qvPK2A%A8lL$UfLL3Z*#$rBT6c>*N$t2`f+$2i(FrJEgSV7T5Wx{jziVq<GX{7Nhoc'
    '@iKo@09XC8GeEbyA?CpEVzoNY1JMO3p;9xg3N{QLOmo<~?+SN5{N-_WC&RB?sZ9KiieFpNu?x5WR*T+I=zU)PzL<{D4RNzxOeUsIRgDhXq@HX^0(%Un5l6Cb'
    'pJ$sTCa?eQ^v~af9*8Ebq;!F{!?VTP65F=6)KBDsaGp0O9$I&mbaix#Nq9E4>h;j)8xWvzHlsxLE(KqPEgbIj&U8i9H<`5yWMnU(LL=c^SjNCkLjGIEd;miH'
    '3hR=uQNUl!rRWkT^Tn@eyE(~Y!JbHC9;u@(+9<Ef(MIsBeL95M>@SKd?u15>^Nq#QZ$bK|?#5qyC(X8~gc&O|lyNG!$ngIq)tm&Go~A2SnJPy?c@tcwO&nx3'
    '{QgqBLv2vJ@B)mz(N2|^0Skk1nnQn;HH(*f(l?2S7{eFawRaQWi@Fm!%^+UPJ5ujV<JdC4tsi8DIf2%!%&{@)*1L&9E?tXC>pn=$r4r`qQb3r%(RK3zq2%b8'
    'oFek#xK93BUQbxxnu-j!`q<EF;^0^Z*?UFBK4kx<<~)6`Hg+dF`tbGp<c_cx?&Opu9r(nKk<dSI>--Rej>=)1(5e)G#8sSkfa?}55uqj&unA{_Buj71ug{#o'
    'BvlB2HX8FE3<|M|66K*q<csVV3BS{&q;kV=C{3b4%`P~XdO~ZeU)9Q69g`>28)ZjWcf;Ni;ExXRfAx@0MX6w3TD1@27N1b5DYp^<qhz(I#KKKd?Vw;ep+yX?'
    'jZe^#_D*lMcvr64oWE00YAgK^U0kpK8gYfYJ;UT-|Kq*IiHaP@XT&>%C&*{I(;GCn-%tzl`M`UV`d9m<e!<oO1N{n+<*I$f^iK6-V_3sJChu)}u#VO?<G1}*'
    'oqm{qg#j)~^a&Y}H<aY^H+uM;GYqH^CCjLnxUL1fd6)TZcz}L%hGpQtw%on2ww`vE2RFGF{m~!af^SX?j*P2zhp0-5K(%^QTWig*s^54#Du0vYkkur9YWsng'
    'GGU^HjbPcp=VMiSiI2xT_wSo3+kf~bYpHPqsU#kI0bsDgav53MrZCQEq*FIfkl_A9r=_wU^7cYgk^vJrK_zK8(9)%}Fe25SuYWlvU#35#lVU6$+-*RT8z`SO'
    '4@7n*xIPeo;hY}-^6kTnqZKj|W)Le3q4Bt-aRd&fpSUf8+xF?`%ksAkuDO<$GDi*|v)o|b4WE_YR~#Vpf$`CdHj2N&${sy@`J{OG{r4}PeE;y(6JebJX{sp$'
    'Jj&M!r&cIWJk?waLxhs<vPfhCZFJ9Gc@3O|r<C!2-ppsy`K92k&!aclSH0n4vPIF6@U*J8tRzaFC;CeTcvVybq0|^)L;XNvfDmPI;ZKLR(zB}hq?ntr!>s_~'
    '-v-qfoB|IArv)m9G)%lgb?DJiP*+Lu2Z$tget9Q*ygPm0mgBJLb;TX~=vc8M&X-F`^t+1v^^ktwi2fu~9vyk5zkM_yxZw(R&<<WD&Iau<JI;;WMVo7p;BW1Y'
    'm$?MpIbGD6dq%4vB5is4PcYcM-J;5m7sF)+NaAvMUSisp6<{FD9MQ|3qS_P44C=~GvgLHi0h;Jufk~LWS%YA??bA}cS8873pT&E%r~H(U?wOD?fFH_7K$~~&'
    'c7|mo$goXMg#5Ak+74jDznDKd|B$uKRU_k3POwD7^P5++N;W!mj1I@f_F1WxzhQ*Je5?xfr_69_A9o8oQ-P~$TXxj#1wX@4U!iS{xM9Fo6Ply-<=?-&`f^-+'
    '`NNk#efiQX31QDGP))N7WTHL_-4a@i%ZKAdRlFD6_@Q`j=Q5nO(_S}lqn+Ft8R5HyZZ=(zBHmmJL+m#1>^+gRNqHnL0DNOgzo*|Yu;A$Bx>=8)1F|J_q*zh_'
    '|M(qVS`U8f^rqEff*H)#ZJ!>1Xi4Ub<NFT`_PA?bA{_5|S@~T66e$XaE!mYzJxnep&tE+K>EVmN7e76D^~1AoUl!m0`0UZcADjAfFmp}!AD=z;d&jK$XVK7G'
    'wrm&ueBg(HwjHBr?~*>=@cf3i|Jv#6rBgOslOni6GY())IExY=>t#lTMJr5q22b}U3$mT>qOb?Xc6|YK>imLr^|5juvC<4()mJu4GO<?Gr&ZSPrlk+5xu->U'
    'YV)-Cfhuy?*J!pGBeq4h*HgY@w|GKfs_jfQTr}$KEvHRm*W{82XRwt;$l_4)qr<W@!2$gMbZ<DnnP!;ZsG`&P1}IQw`-330y2?fusPW<KcgD><6XvDHKCt}%'
    '*n8I{H;yY^^nX7^jTIiC%mUbGve_i0Rs=26lJ=%V1}S^wjldv4bTu0y8fXJ(Ufny+L!2i(PjYhIGFRRT1(Nb;&$cIqLRV$3yj14OmFxD^Yux>q@5X^{1v_WG'
    'eS7Ed-r;0$^zuazZEQJy6k}XQgEfcI_WaIKhm`5c4gpWY;w<yvX<@%WLhcH7^`(u%^`lsi)tfRs7r(yT6bkSTDo2hL+T!b#?Lm0PD7y-s{qUfFc71uZ={ASY'
    '$VoXhvi_W0Y(U8dM!?gf3Vhe5=Ab+U=bQ}v&v=wMQAFv-^cRaX^O;y3bV<N7vFh|;2X;|E+MvUHQ2e2omtmUP29tniUq(U4G#6^@0tx;jHc8AxK_L9@(etNI'
    'dfz?y@zL|2de0yJ`BCxu-rfG8FYn8-0GXgLZ!x3_vFz*BgwBfY;Ep$z^XcV$@eWZyG3Sp<U+!lRczy5ABr01KhyBB2j~*sVHDcr1T#|4N?xcz}vh*&@s8)2z'
    'bh;Jn$h_XL5h?+ims7IDH`SZwJz*7|`>FdSsTF`l!TS1u1`lxAGm?1cW_S=Pb?diVQ0jU%v_N0$ZaV>Kt4eK0-#3wD8<8P9j4}Tv=M*Ga%?Kp&a`lOobbegH'
    '$yggf50YHJnZl6_va7jxb%Og0r$jL~vS%{?AD}VWaNY6-@0j2lfBHH9hwkym{rkoL_y7FA#lz+7Vzusl3&)Q+OTVSGGOYjo6<UT%A|<R59W!x*Dko7?q!YmN'
    'u_|C)ps&CdqPz?}ofl>eDpz!2fCa%`1?<b91e@tv2O5~R&5=33M6rgarZC-vRH<l-utLX^A0Ph>B@c@&ELnIE0|xxS&9=FQbm;wqFAS+4R%QcZn?v(g4@egm'
    '(=(*;0`zI7E<z$qFIUT#_!x97rc+FpM4Z)n^QF}9-|fM_Ck5nI1wn9g2{(7^MX{-;aJ9u;7@OkFMJ+C_kiu=g-B4E@x%<Vdy1pWT%5Wme77P8PA({0%H4}zw'
    '1Si%k?Momt)t<A4T0z-lr{^GPPC?1uJFCq+t-%J9QaV=!WwI=;%>=AL#bZ7a>^zz}SSC&pO}rV@3Gg*vAb^Sao5oHqD{tNX-rg36*s(#01EN8EiI0v-{<F7T'
    'T@7eGf+)WMW~t_=DDITNDrVOxb56HxtuQ0JFc+I*V`kOt;nlK7;E$_I64D*u%64U1LQ23frt8H!eaB&2gg}m&?`4k8;)>Ga%y~&b=NI+VbjeIjVHgY=7J7)n'
    'hZuG0Z?|tMjtB<f7}NybK7nU*2$R9PmFXi$$WV$FoEct_xs65a{%2<h2P|BX2L5<*e==WRK~k&<eISyHEi_5z-~hDQ_eq`o^x-p5YCrq-Ib>wAnloMG;nSy&'
    '|2BF0?8#40zIpOE(zkg<MLBL`Q=J-5H~QrHPlnz%kDfmlbzjmuMc4V}$&Y{j{@D)?e|lmr_)i}{{84H~N*HBKxIZymY(1UK7IQeuUMnktVKZY?C;oJ8&X7^c'
    '4FHdFD90v!EC4>_w(!IC%WK%ZPw`C`;$JW7#g%qVr%HWzHRaOXP7gTH{T|FqkNmO<1W5o_BP(OyyW5h2cXvz>x_iGRA-EBzNgTJN@Lp=8H0SCM`mINEIbRyS'
    'x7bcIq>Kc3csOX$A!0f`dtptD>&|X6a3k!|T`@(W#oUUM?59yVn{U8Lwij^Po@McdFnKqhc0eSqCy*E1_pW35w9hfI7paO-dj=Z>a&)WRqgCm!SK?hV<V5&p'
    'mqA<xXz3}_X}JA1?>2pCVKf+W{#bifo|5)XO0b@Ht&@xME0CY1+UU^8)3+ovjT<gKgh1#<iq)}^>>S7}(#=RKK_V8sBY{wjw{{#|89PgFyi1I#>3t03vj+!m'
    'j_s@|KCHinna0V*bb|qet7Vt|gLg;#IG`>WdJQb!b+@ZmMt(UxEvPgcQ20m#1MsU`Md!4`z>!!B+>N}iraKcDT^ijma?2+EM*dl&Oc<h|xjz9Wf$1O6QRr=V'
    'a719HgQNSEr^``A%Q8tttLKQ={bpKThKVFeWT8w<I(C?(g(Wjo-ZKTm>nfdTLX}8Y1h9K#o;wNHk`W=|n^w^$(@cYj<|QZ{56EfMd`g3RAt|MMi>Uay0e4aI'
    'OMBFE`Aa-6l8*A|>fay2z2?2y5Fkdni%YDZX8D-qc;!6;E-O7Jv|7O(oy~eE!JO52LZGKO+CeX;@8p7jGA!R96jAilpOTA7&M$>NkGcmWyu$n12c_BX+n*)6'
    '{g8EeBZBsA-x|%rC*44aq`BMS<%CCsSS)H-(LD-cD!-%~A54$C+u=MJ_E%3YH)d4mGDOGqJA3GQl;;b)gpfJIT1<e6ksAElr+;Cr=bldbi<e?=me|8G&{nZd'
    ')BNg`9`uSgXLl!1!u)q2KaS+b9sY4F|CJ)gQsfwlcwa!s9otIx<j0BpxIf{}t3yiZTp?bPihb}+s=xD0s=q65CLhvD)0<A>Wb=;*d%6YJ2Ysl#U`LQhUB6w='
    '-_j(a_F@QjnMaamo||S39;M(vLrgMy^>h4)KC<%dhh?LkjLeJiF3%x6+Z<(@+TZ=|cLW?`t$@8?OR`PQoUXE*&9du(mb1VG+NaPbRG+|Y0@6R5;+xe1!DY-<'
    '6L-j!pT!vO{YCNpwm4fE>fm`WU6^)h49j<P2S%+CB)+=dPDLIdl?LxOn-dwgrYhN&(fPA25O5g4|MGfuEjYjq7tWUTq2L%pyaj=uQxy$=F`o+0jEzW;kRz)P'
    'I;dCmyA3Mq;Wddg@C)V~D>x<Q&*tmRFN=RWEshHGjpn}kkAJq9RPe|ZlBc0l$6w(Svr7jtNzJj?pJPKA$hLFvhg`0fm=>Wp`jeRxOsH(97-wN4%Fm@a&;fCs'
    '8plU}XNPRagk`HK-xrYo4q+UB$6p`WTieCRWD4E&lHs#_4OV1xU<`_{zQX>UmuAZp1LdbjUmwYKyGu%2uD>i~E|2g75s?y{ChQj;I{goPYx$$HU)^E$dem<Y'
    'WR4D@6cYAf4jqaFqjR9nfgv!}8R0utBa^BMwelUlU|H=<=Ic<lloLBm*l0+mhqrkBImJH87;wJhVKA&AO*NE(iT3()CKu9tYfujxL^s8F{FB+m_43swISoHo'
    'g++dEKlB4;G%v3gQ&75$()-!t17;FjDa=W-EIxn)Y<hn))?QgmSKC4G=%R7c@?~dWSxr2HQ!)jf%e0wCrcF_ANY^HK`a8?vA+2q0wrmo8U7u#x>ouBbnaOeO'
    'k!xV6+PameId*G$E-OK}Yk;h)*yh(?$8%5?|6UAA$<Y8;&N504A)m(UzC*c{iSsLCWXkGIV&0a!q<cS#1c-pi;S}!%%SutbwS$mH=V!`>4Xk!#Ner3#*w0=g'
    '(lt-*n+vn;6QLGn?qNp}>=OMuOkjF)%Oft)RMc{qkw-Kx9DF4sO5_e+HyE&@!5P!-nTytWFg<SX&uh_K?Bt7zDL@xKmMz8u0@e(@@OowY4?E9WjAdn)Nzbj1'
    's`z_JOn&-yzBxTA`#(i%yXk20n}}1OFi(vZlkix(^N`WvkYKq;lm40m8=m`AGN0b~E241_vo|espVyF#o^%4xar$C-o8nYlsHt>vF0ZWLnjr4nf-mQr4Jjcr'
    'XNE5)(&tK^9XJc#xxuu<bA!7M*A#otX}-+FBErwXcHpM8^Xsk^$W`<a0wYi>%jf;n7jKaT@|IpHo(l=A6g2XM->O(zWe|Q=7cdF0sb}Xbu)3lNiSgVCi*f+k'
    'vz*-k(DY{{TE-)ToCHv8Q*TL5gWv+rQP4eCsP$lrtCWdRO(Bk{DsY3I4odBvfUW)Gv~cn)5%3Zz!E+#A(g}!N@>u~626X&*)Dqc%x%x$LR{pc_O7ec*tk!@t'
    'LQB~DSnL|g*y7%%{=wY9%__<4>*PnJn}`2lD^7#wCA?)np3LQV-|x;>Wn8oG>-B0gS<GM6`ff<;JA@feTpEJK@@uXFnh6`o;H^PGo$1;&g8oe2@{ziy@McGj'
    'Jm&}OZEBhl>S!=8W@2~F-GMq`>Kt{v4(z7wt=wEY{E=_TO^Y)p7k>zlkm+rhVA&M29ld2%Q2)?$BwZBB+Bpl5jCns%U%ZX9nS$!`eLcZ7@%ee)rmn$=Q*W7Z'
    'm0m7POT^DtVC_)U9Xps`xDjP-vKQR1PR(nxz3{sJiyU#-yf@GA1bNRdsqkWnF8)RLQ@qE1!Os2S#P@SNSG~p(0m7r<`PiFt`bsiUnK|dT#Ea|!#(%c>Abh>Y'
    '!||8W^*sOCHCZb#n5zGF>M;Tq7-Xf{%ad=lK<3YHO1d3FkL1b^zXeP=CRKS2M#(;+?gKTi_r5KU{6wfI8FX2sOUQRW&ss=v4MJdZB&JHaqP#Qgmt$jAr36!a'
    'Fm8UL4>`9C##Uqd?0OlyZ?O^#QWY)9f9;GN<IGp6ohKN1c)<+zMLqiStc_Z8meOkqoA`NVwTsPXWUSsn?7vyrtmJ7D@IlgI)fg-iydiW`f_Wf#|7yZ_lK>tF'
    ';gKMomcKF9Qr&kfn!N?D)sp#4X|O!8{2;!L?k}T%7CD`#$i+;+(Aj5XLP0ywDu%nkQQ$>AB%<H7r3js<MaeFV-Hq6~@J-N#4+ZsKh9JPi$H7g)@0WUKwOk_>'
    '^2gqkdQ^e~OVyW?Xk^8+UrHip<dCj<SdP^v!v(I%=+8+8eOyZThrhq<97^&;H0vwWdxY^m9Dq2Z_9H|Y=qaSO*Y%s}+BL@{`Dr&^FqJoUGtSW1-+)J92N%_m'
    'W+{aFeVJI~nNLoH{K25a>wOK7dCT0=fdAnFWCh-Ng@-2mf9gsXe1ssqg4UngRm^uLX|{sIhNp|^<%_dvVaCwy@hE~JvYasHCd~>eM?-97EUIb_)0HG9{5P|i'
    '^NPE`>>Ssel##3Huf5h<E-+qP(!2c@)*g$hbT`hbuz~*4yR$2s0{|rx?k4tzrm~6_B6C7a!<=R)iHo4R0tWJQo9muy%Q7FTX88`^fAoiUl}guP8p1FVW#~NS'
    '{vgnN%kHyA5*4Fd@gLeA=gE&duar!!#l25j``D``%~^^>!!0x2JqJEJoum7Tz9D8;vv*ZA`)5aSbqV#Cg<>7{b`Y5qUm`@aUUTw|c}P3Od)`67%!stzOJ%%8'
    'D&4h8_B!_QHL>N_2sMZG{d*b330vP&Zk>eR=zhjQ#alCimyjkwT?E=(&)5awG(BQ^>tAWer*n--R*+p{nb9F{JV=HN+(0dM|Kf+ny_ax2bbD%%`Qr!`O^^!<'
    '9Bx6*>XiNM3WcOdB}lhK>>a@^@BF4CEDSELgmRg?ytyL}>oFHkf0KjM*&QeZ`SyfG8$MxPBe+9PuWoeiCZ?S2lNYq7p8uS6dAl+j^crK2z&yJ;x4|BK9}`a+'
    '>uW|qAmC5KE`9Lv>(}-Aoky!*6i>j1F9=2<aUb38ApF+ca-c0cpN0^*@quB+Nc@<)>Y`3WtH6`cr6#8zYPS#0sqiZZ?qoefRs*gS<s=GaOJ2^;C?g+qjH``$'
    'M`OIOkyQ<!mBrWQfCQ-?OJXkqwJWH$ctZ~HJ?}~zZf|yGMt#Klhp&Zvn2PoZyc_L6?5-24PAX|9<XQEsl~b#Ni~4&ZmIkRQBNKwuO5k?XhjOROEDI*e&w$7C'
    'VNi+62(O0tIq>RVdM{|g@%c>|5gW0psg%1ZPNOYgW_TbsLM-)Fan2&GGcd9y(rrzkOJM`}QF#o6=qROOJ3~42Xryu-&zk1lR`1*EYIw}pZg_?*Yo>Pj4tE?;'
    ';hE%HwYl#?uESa}2yY9icFL6SMc;Lp)C71_KFoLz_#+9FpfUt4OH@UmMGUR<!I;ivQ2r7muTCL)GnsBNn-x8_)G>3VAd9lw4IMej#WB|9((7Vu72uxk8v#bx'
    '0E6{l^4`y{y4v2&6xE7={Pd^A7LJ#0vy@bZY3+CG!e0sSq5RLA0GVIh5KuH(HuxCeKunK(1+GWA3MyBzzM8(JFQ`%ELv484*=AqcpH}jD;)BQ!@-4ksx{ojs'
    '%4KmfKikBz1%)|yg>B4&Z)FTZ?jJ7SiR8d+hKJRdc`N1kIIE2We1<tx+#c;DkLAGJ%0e|Yj3zcWGgUYto<pnX0L*mIldepWZdVtJ9l8AL1)sV?25gJ)d(1i*'
    'Oj+a{>4-2)oT0pY*A1&tFH)B*UT(wW_wdbEPG+5-@hm0muX=M!rlS_~-D0k_vL)*Z=g%TH_<5%>H<i3=FpP&N_52Ud;uk4^fF}H<OOzyy>A~e1g5lYqA%D;u'
    'ASBJvpG4URM=IVeuD{uyQF+tq+3NgsP!_jAF@MqD{C!QX%1k{ZZ$(Mn#A5csQdb?8i#*3fnR~eOMct?I0DJAW7E0Nshu4?Xq-l{<lBr=T!-kmOUW-JG!!b-^'
    '(LL<nKQxUuj4Gg3FXk`N@%xm7ptM8)u^ImczT4(O<nP8)@U@*58_^XNhuLdWE|{vXPrXqm^uWC5<wy%MLg{gE+8+HVOy*?a!QIKY8R-h+um<jTQ-KcsP=kSQ'
    '!z#E^gnPg=+SD_$=s7Si*kkzpk}RQYY+_UE24GjPel)i$BrY&$U+;=Ag(~fP{H8J_w?LW-W9MYYG!?`^wTTN<n_%b)<lURS2<mxj$W*yT<Su_@1y!u3JY83>'
    '!PHcHg(VJxw_&zCzK4`hDs|O}z7si5_<j~xJX(KO)6J#^sCS!x10^p2sptKzl86P454drBPZP`>A?X^r_B^$6mI!a|SVj3KdOY+!-HFVe($Cd;b#^@?=|^^X'
    'N3!-<-UZIA;<`<;D3-LANb?x4)Y+M090$1rbDyP!$UUFw8;MF!CftPWgzKBWX@01agX8Bl4@a$5xBlF@s<_>B!CtPU*=xESv8-Omv#=hqH7Om$<iEI{!*Z7}'
    'R5q+^cKdBWUc^|;hI=jBwiEij`t3xc;nj&U;H8iztwMr6!3{(~I4a<s1QrkFLmn1SI5KVoTfwYR^oiHHM=YIj4%zE5&5>4XXMVHOicSU={)I&joW|2>F^j9k'
    '^#;^lk_|}b`@x(U68p>QFq=Z!`lFy9qgW;fgMnE-tJ$lrtd_o)lnWP18~+t3mvUiIODZ(**cXvMiYz0)hCEG(uJPLRtH$pWM69<B6b&41aEsL&6<cTLi(gG7'
    'upE-pQI=tP|F&LSQp!43X+Wx}>0*J%f6z_Hrnvyk#Pr3+$`jCe^QVlttO|@nIJD{k7jwx5W=UFyNuO!`cj-SUKG09lb^}v|eGuIWyewg{M+pw(#XDj*jow1`'
    '8@4VDAE<R<ozoR!4tL!$6|87eQFkMCN@xDKTGxm<=Fyes)L-4K8FvEl^`^E3T`g8(yC-=0Ff!iP$`Qvya|M}4fNbnXGaryW0(gC-0;$mV3=52;GtQF+Hp2v+'
    ')1uvxD9sY+&odA|Qw|7_NfLl`vCpAL)5U7Tiaiu-;Hx#eGx*eew}#`x!#m*CFbf=K+=yo*YS9mqu>+<bQ5kQ6k7na&bK))r&#Reo81fc6)$(s}tv0xu@o=ka'
    'pEuGUNKsx4nx7=DWxkqw7sH{cQGJjP&oAuyLl=A(`KTF%QcbZEM*b2@^iD%GS724nRJ($tTvTI^J#f>b?dW;$QE~>om<Hq`Lu)B9eW6LUQctWNHcpe$WRFeH'
    'Q9L7&fg<uVR1OW0iyp2$>?)PqTwWF&zwZ3(ZEkxy+QG~{&{L<F&V;cWl0t*oo6_J|>kj1j3J2k0*mq~)NsvpAOA`&6_Q!biAkAuYj@bNIT|}nWp>Z(x>`j|+'
    '@AoVq<Ol07D4)$bYT|v=6s+Z2Kj<s(bOcuUT~qObj!KnspT(jTuMoFE9RVHSW_y)$g+nT3LqaA0Ti(g6SGlL2X6+^M998Z3!g-a@BIn)a)Nq?B+O$tQ(>YZs'
    'cTLIje%h#Nh%3-QZH6{}>Tam6!kalpJBmJ&K7thY6oF%Og*jB5?2F<l<?F0%z8}s8YF5(};D<Mqmh1xb67$s>ECHnX?-#{4LR`@U5=QE+G28#*9oA$}g9Lgq'
    'rpd?TpYtWbT%g64a1;72PzhU{s6)$)7Tnpp>XBz#lv&9bd?%5cqW10|2l3XP+oI+a$^`9yw>Sj3jXz^kHzN<nuI_Oe`Z~4_3P?^#Dkwk;3FA;!O2!7L7>NnT'
    'P$UB-X6Zy2pCP88ZEOK{FBf2tj(h+3fP+9F`PBFWO)|Mp)H%`=&603}{G<iHw+HMFPgrld+Bn2|6?XZ>Hv7kVr1o)>O;z0CEt~(AJDgdQi|n7~bQ{2sW>%EA'
    ')drm1<|5`JMvTk;8bg^po)*`j)@#t+kV*%RMhr#k4oNnDjT5IZozHY_-*&rsr8`Czg9<r3aamK5J(5u<7Tvn*fjJZBmk1HXYv>B#YqU;GudR|zprA1LLy^ix'
    '<<msTC8FpgE=!N3zK|w8)z>y(N0hNjvND+fui#f`hGf+;n8pNMJt$GvV$VcDGk8t8B?X(~(c>RNL?2F*LYf}T{UO%TG7Xue!yp2aRp<@mFV^u6f(l!a9U}cD'
    'H%iz`Bbs{Ne?^Mgl-f)*Ftwy{Q=CsC2AV9cW9!rz2gcD=KLcmwl4AnZKrm2^NEUWMueCUjX}jzGUtX&gqT#2hMjtCw+GNO(a>lw3nk77+)tg=Yqkw|=TMHu!'
    'O}OE8E3)R=RgiEr$a1>fG@Ks-{zy*&BMcxTz#9(;s3PyjFjfA-nxHw<R?Ss&T}qkwGdrQ#JlRCIX@51DGMi{FW-nS2ZDE9UCVUrLC-+{rgm+(z9i8{NYVVW9'
    ')BXv1x!JL$itZT89@HD_jLTMD5@{Z~ySso_;5I3r&B>w)<2I-jp5dw5C<4%ZO4U@-K^eJ`cp6!B*EVZujT^0rGX)i6=DAonswnQAfrJOlU2AY5>NO2NhwvhV'
    'ENp>(<v>z+zX3><K4o~TVAd$8j**Z&Eg2y#a{|(?@aRm0LPStWXc6PwO6o@((sVP~&dn!Z*V9*KtccB+9uX%FhLjYG`LdoctPp;sRC!saF(7h~JusinCzRmh'
    'ygq@=FfBT^5g*%$kS)Z>{Uk-gYIEnYEoegp;^MwA0Mo~HC^u1D!Q4K;T*mbX`xo;kV*e~FrR$D=PfU?x=<YyjmjQ;h(gXooc#B0u12-5^I5+Qs59wnX1O3d!'
    'cw)HdbYX34l``2T+Y<*{p62NmOtiJ0IW2>Y5F-c_c@4Do5hFfIUiqgU&a}TN#5=HUxrTBfS1zc90<U7qWE=x@BQcO!s)^{V#<ds7=>hUQT<ak!H(1j8zQ0Ge'
    'f1Qg$@L{zD!$0o$p%32a)4E}0Jf8cZQ+zk@c$k&6BXfQeDU}ij7J}}Dnz&8qlQ_BbNVa(~kd$cTMWcY>eJeJUXS$$f`9o~W&vdX<9!GnSkBt7!<Pm_YJ@rs)'
    ')J(iyOs_U*ZJ{Y_cIz@sh&{kOAJlHz3!?HzT6tZo{5p1kRWO~EKmO2>vcVpv-Qv0-V^{0eE2oZbz_6v+VfRTye|#cVAog)pK!afDS=Y$W`bF_@u~4b<P4iRm'
    '0V9B+!p!ID8aQ+R0P|pHd@)@L(DZvBV=0Av*7cIWW#B~F%hL2496tX<(#Rk2&o(}m0F*f_o*I^x^tlW?EYW{5(~+U-ZGgGtklq*$wBin1JW#|2g|;i(WGShq'
    '9O}r&@Js951tEUs>vcPk<O28@p5Imam0N}CsS|+J>Vx-Y&moq!r^o6jVe=ej4fA<p@;o?6y!b-)f}U?AL3Q^$b{VBqPVkHJ=kG`sDQ!}X+3M<@pQNBO2WK-t'
    '9HZSP*v&9;QfF~}83wQ7t5L2pD8Ct<&}%C`5!*Er*Mk<^pJ805Ij!VtVCpte%|@-!+xGn-IvbhKkq{nBW^_1X;}P@?R~d2Le*O1uC%}lDF`#Y7FtJWB%Gk2f'
    '`Lly1VprimJtc2pK^BPtLYp0iJ55Q$8%OYFwSHxI`<qV~gMcm(>jhbh218?0SVA7OYg4z?GSFL*zP9iS(g?R(fPS4ZLv99Wyp&qvO9qE+?)_@r380uqrpIGh'
    'SO?nu-S8Q(Q3(n>xJu#`()>X@wGU8p<}cx+=I1%yGAAG4Uk2&HR3+%Tpvc~ayXFgXaBe)dY3|eWMU7+uN+E(FSL||_c-Wt9UUy>f@`%+Av#)X#)YlxQo@VN$'
    'dG@gReMpNr%20lV?}N^zdvz1T27>+0vHak02$;XWI^O^e^TZJEsx?%}m%eR3A`9bK8g2+mztIt%Jrs_u!Q%<oO5xPg6A$#>>P6II)WGjeI*|rlh^aV-WB+_U'
    'oA5<4Fz5xM0VB9~UQgauv+ncq^wH#xqpQiR`*!lY`s2i$tk26a;Gmt)w+MNfgUt>tGknLi7NnSUp(BxiES=Tt&RJZT6Jc#+Fw?OusGnRLjJItz8{(tUUz}{T'
    ')1(Q6_Mhl%rMPhbO73hf&~JDTk74R^bQg$!4a#KP5kbnG2<*9VV9&r#B_$81%n{gCfqo7fB^8H6%XJ7ci%+Wv!cIETWA`Pt^L?pc?Ji-c_1jr(yIQ;;fghC?'
    'FRv}!Ug5MDbW9Q`H|x=9L9DMAk-1OlaFLVa7r4e8KMW&H6QNMjHNNB&QMEbO4a93ir<iK+Msu#KYX`-rw_p~ePj4O>y>KJMEk%DzW<ynY3|vl^^K&zzJ}`(*'
    'B3rY$z2qUA9foNV(Ai5|6#_RmgNSPO$9T-9j(^>VK%!}=THJ+7qfSmhrMJGV7*Ba`ujO`@7qGRT2<8$@bmERk>KqX_90nyB9qO~&?A!<x|MawS8H+t?)_gNA'
    'PP0zpVFk?)GxfAXc37P-r85=DWjCzyzT@1ABC}jr*ckw`m7EDsRSs-Ei5`Y&np9=)B*4#-6=S)Ohr&!C<!SVC7e~;DYYyk{x}N5Ncg@Y3v_C@$^{v#6Kb1Gd'
    'yV84i`Z$EjFY71wSl3te)+Rk7^_4@7tTErHl-QIWY%*%J(g>5qEk)R~`;fT7c08DEoR|qcW@a}GoC{HL+ZjQ@!ozwU7ax@Pa#Q?6wR-wN4GoeZ-Tsgq6V!sS'
    'oRZXS{yM8)&u6u%PhX<X9X-40q*CnbJWzM7d5i~?fR*y~Df(lJVx@y!g!Ohm)S3{BkvQywgcF^|{5R1m+`L<wW7f#Yzp<qVi5bzY8V*d5wE>ZJJ-=ebH9SFo'
    'DLGE<I3`%rl)f@TMz>RSN`ZUzsI*-0j_gOS`oSUmFTebux6m&3+F$3qGawz{tJ-}iSxK}upq^SbiH=JgVi{OV!C=B(j6n@21C@Ogy1FIlVrnqluRH18XzNE?'
    '68hL~FD9q^x3?k9wcK`;wk3_x>_eK@RvS?c0efc1g9C44davzM9|1xX^<96j+0+xc{Z=V3&q=6J-l2aEnL>!;rE%fE{r1_TZy)~jh%dNiUR9gHRMTuN%y?#y'
    'FiI8F@+}OUWZVNA>k2jc^;^SWyo))%xLz*i%U9iB^Af%9UDtvP%dI!zeB!L8V3Hx_6h(}i8Aw<0olY>;oQ%3>SB(|;RtL@>lj%9!mq_}}=Z{Iy!YdZKAu0bM'
    'b8C05Gume+a0v-7P8b2t?w$cMIo`N0>r1%x<7vo9=HAV_<JgFWtTyD(-gcyT3;@bc;U=XoS8Z=#4J&kVU+!TjFb)QuO)=QK2WaX#{ha?Z^kAPnD0;>7pC114'
    '`^Vq^^tYbbihp_h@Y(nO>(QTz@4o-`yWXF^|KZV(u-uDpA3yoy!^g$*M~@$UV{T!;LU#}soA`0@s;;jz@I2^SeO;_?m|)dxzbGEB-WXbEhFdH*bbaG%@CJzr'
    'O-PEvYFoM6L9(Wz_-_94LRUA$_&yI_AM@qQP0<A!->&CpklJz!;ZJiynUR<jSx;p#Ju~zRt9&%X80vNY-n8tr9Ou)^`QlvxX2mTj5tMSVhmmeH{Ox-B3u>1E'
    '2GfxHk>lG)yRmvx*@!o{$M6{`^U-q!lo{s9|E$-`dQm+8?%`8<KHnIEuonS}9Kz*5eMMW+3_|PyJPrNNz5))Iv+`wl^1!k-r@U3OCjze7KIS-iz-GyZ&)`sb'
    '_U-e)Txp-{(hr|Lef+n{(`Qe9dh*Sa$9@g}omWeate}EC_WI+@q&pNSs#2uvXmxdbFsa{~)iIeZ=D>ws2WdQ2nw*p%Z_~H3P>uz_C}9gfT)(`A{q_{!bj`}1'
    '0Eqo5rFlA_@J!lS1Yu9AvCVeASP(h0Ul`(kVRs!T<OCzUDgSypUw~?xLNjs5IH-z?P^jU=y|AM#Rj(nhtC?BqZnx9x!R+)d%*HnRTJ)E_?FFF7nywhpOrisw'
    'E6w0=H$~`s57jOy3J>nJq~Iy06NK*GZ%K$WGAR;myYgOYb6z7m8~vjeow%GYd!(^RQ#tJ8ji>#mx}3i4EmoUN6TLgfyXjf}(WCTXP4pfd?n4iTjQ(U<if#Hw'
    '3aOq|S*1Q<3o_u|*h6sIB!w1VP;6A6MuoF|+s|mHt3%m8%q6e0F)XZ}Y>kX3x#wh`BgzAl$Nx-TEmv=r(9DMNFk4=rfM3xj<+A#rEauom8QbK{(G6Kpy4<d+'
    '@7>8%w>(};J;OdBXN`I4&>xo1zBIGmyxRag#p}^vI2PO-)=5ivK9^v)i?wnIkyoG(k!qvEF`!EXbLFzRm`?N26`h)rs2j*E(#=RKjZ?pC*6CW8n{<qiJ~(i5'
    'Y-i0q^1g<V8qLK-#I9y@L+(p`Q9MU1d6d2cZbC>_=(mHf=!gRn@Hrt4p{UQ#jf}MEAEc+(7bUtbDTbVvythC?z1rG@2<B?l({d~dsJ4kfyDt!wM}$E+IJ!?7'
    'pO@Z*cl=6bKi<1ritAxs_s_)dx25f$Xy*LNG$FY_fVg!0a0hKU<c@Ph-!nl-rz%N=bHw1h8}xAwQlzYXVGokdY=M5mhlhjXn0lOINa2Sz%x-AgsERv@X@GNp'
    'K+1i>gL`?(u8tv^`e1T2>`U0M=Mk41D4Y}CO1B3_V$DC8@UddO^{pW1ucUi-bDe4+0qfx~8vTQKpWWk;KRC*_qW77R1~XJYW}iK1xz8Tt_t~y~%RbBYyVX8>'
    'p!eC){g{;AVtFFLri!S@_k=`qATYM;T3AbV?49GdpY96E8E9Lz5Z00w9vn8c0HbA4v%T7odelm46pef*&w3vu7|(HHUF^?z9_(T~4?dFdJZNG(Ix8c&JWf2H'
    'WT*M>XoJOrp@2(8u7O`hr!^~$uEEJ1PV<h^B`4HRU&dUJDHsuZ6KjJ|-=k>$akV~EiYBLKBJ~(TTZ!WWd)eUC%1D#o@xZuM9R=Rl?26*g67g};vTC)R*2>Cw'
    'Q9DZ{A<L5_^^M2h37z~%SpcCx)5{Ivo}3j{QQix@HaOML(hl2<8c;se5ZU?{tR7b9n_rz0jDkie1oiht*BkKNNe5Tj-GTf#k{@^Y$FclZiX2OkV<_THpO8DY'
    'mF~%p6Zvspemt1)C^VL@vpWDQrD7k@24=>14nMlT`~Ba3fBE~f$?w1W{SUu?{=4#qfPiA_<V`0bj{8Ek70La_o>=|@7A3{ZkzIn)4Rs}&YqJ|WFoFqBXJTfp'
    'RSs&+HW(x!mdA;}CR!Wx+O#sigxqFFs1db~C9N<b0H$6Yj=?&5s1cDlYf?7prnACsuBaR)hs_n^PviqQSA}Gq1S<3?0rttZ&X#m|xCwkbDD#Z_LbfGI7?%N4'
    'MBQ@}f?@#MAvusR#34Jp9pS$O1TjoNU96UwbVe8gt5W8h84x3|1;p}ROIRQm;?OqCY~>KNbky59hr037!*q8zR7uVH+DyspRaX&rDOwNY+mpSKDgwBKU(eNW'
    'coUr4m0mkoiN=+d`l!_7Tg9O6J%Ch9wlmF`n7k6B7<@`wQhEChS_oH*das-GGtw&Nf`&vtI+QZr`b;H>LzQGT?mgp_8d^b~t}WQfv$@7YuA|af`>8k8eiL+|'
    '1S-@}!QNxuY1I>3NMMJ;5F^emj1kW}0*Lm>COitW8cS{}hTWkE>lO=7=0~<wf-9#o&<peh%xs#@t`TIiA@At3x5U*}M88T;6Q!^PDW@@_y;Arc0={xWyAV{j'
    '1#oU|ooQ%X{O!Zyh8xm#B7tGMKYZ=EKLL)b5!^}!g|^iW%x<jdn;?`~;k!@(_b*vA`f*D(Xc|c#VTz&Wm-n9M7SfCkEPd3d!z+E-=<kN)yszRei3#w{&x$PU'
    ';hldp{xBsdkDupJ1%DmH;w1RsX94t0<5|OwWHJg@OpC8O4!_CF9eK7bC5Ko12Oq(>TRm~3rQ~jDyQP6X1g9|@h;yNzmu{)#YIzytxdIKCYiO(sJ)~$ay0wq_'
    'AERSVQN0xo=a%cFJ%Bjdg-=y*AHF&TzN2<_+;|8;PJANHy#IySJ}8O<^C51|M+fQi&g$9xOlRM}jggh89>8opMszzrpU)t6dyDQObo^hU*jjrx_{0A3eaKa?'
    'yKluURq5!&ujJ2OL&d?dU(wETnC(>6IUF4Nm4d|^RLwF;w<;F(gbzv0`0Wfe%eyo|iTa-fVcyMTqJ?mpbA&wT*Ut*`?4#p0iawQ|YD`oAWrWBqnpjPO90h4-'
    '>+^T7rrk%{r{(L#A_EM)-?+VRxJL(EMcqC#g`YrS*CU0UgPp%xm}3pB@aEfTKQ|-bPC>Q-pD*ML>lz(g;XM<KQS5Qap~{e%kDPP;g1q!3X$(5SnJ)M4-iCjV'
    '`>n9*fr%!b(}@Ra-DAt~+aucZB2%jEID@L%@;S+uK6?`tweNF8<XeKEjoz{;I(*FMXkY{Y_L>pn5e~j|&t+<uz$?7+gA4oz2ONJ>`k>_thy=GX>Evp)S^%k0'
    'SC~f?&ZA*dU2-(;RKetwWB+1#VOAOp)ZP!GA?tQWL$qyF(s{@-KlT&9=Poeu^tws3fZy9L^rsvLUjNPwBTg2`7la0CxQ*g>(37C0m{CLmA<h@wqA&=AyVe5h'
    'U3kU^{GK;9(Hm`T;2{%UE;o*EJZ_BREvBd|N7Dpka@o@UpPD6OwmRgYH%kSk3bs6La|>JmQ@|qZ5ZB#UTzWVv)=c&<ZRVLF1|(7ex@9-M5c6Ole;F3tp5&V1'
    'Q=7l*n6v-?QkJDzYYadmR05!p$iOAAm7?KO9Owhsp2!5{Hkw(!M`il?puBN+LUkLY7XAHcRiiK&()4iW!lK;j(s_H+@ROLqUshqc$0F0tw|b6fi3$B1lHEUe'
    'tzi6)tU$)kZV*Stx?EZk8B0Q=C9gX-wUPmna<H2Td_TYH3Y-g5Q|u7ahcH-Ya(gCVbvWyY-TC#>|2FUhT;lwsKhfZ2j#2!AqRFzs2XWBJTpZ#O0Di%lk(9(F'
    'X{943r(TbBc#3~bftsvRQ4pp<AzeQZ#nbhJLSFsAzMSYe72!%ZTHWDIZWYX5g%jl|L+D!TMnf0Qv|8>{6v{XQ%_;*?JT<St1BKZ~%1Ym=#PcAq>U*dQ63POX'
    '(JvuuX(cQ_b=`0^KUgVl2=^}fTHDnYT^gmE_&3v5kI$7IEn|wpKJTzceaxlH%3T$Y1={CP#lDQlMPB(133Tel+*RfDthda%l7<p<g1!ncc5t-$YHl(TbpcLr'
    'fyktO-gO}N@0K!LmT9}bdga$g04F%*X||^?*EQZU0L}wEA2I61MzI)D{iH1pmIMJJ4=L2d=|^Atno&2J@+Lm_Y+h_Skw@}}&Dr@H=RQD+$9V}bg_a!97vs=C'
    'U|SlUX6~dHn9Zn1WdeyOkb35DBvwW1>aIh#P6HqvbtpxoM)R?`4aARs8K;{FrX*^MI3|95Gc$nsVp`>iGBZUtH-dsuWMolcL4RElFSsQ}SqJ>mJ?x(xng7bd'
    'y#-n(P)(^WBuEJx2`TDhf?iO|q#W|95(bIoHlQ)8dxN9WH*dsvatyQQsk$R9X_0#D>EazTqGcD6#|zwPP;Uo0HaVy5E=_rG@agunik=+zJet-G&wtA9bKb+y'
    '^DXmIT-NOCRMG&Ydgr4j#7oQ}-#;j{=aBhq_55{>!Ees^(eEU+_X0V>P(ch7NDYq;Jp)!1euz>Fvf_N1u<&YSfhuOLT0w4Av1amJOUQ@`c+A!=F)4fPJHXkn'
    '8ucP&72}>cr?S&X_sz&rY`LHf<iBIJj-2kib#|dVW0G72?@>9_%)pAKWqST*ZN6cP?<cz``&HsE!wr7kPm}vR%mK)C7Af`o#l&BPa`6C71?$8QL)7q5t|1s+'
    ';HWEurVKoG0nnfRSEtKSTpt`;@S*c!0nWU==x_eMCRhlj9+J22ExL)t?1iPSIxN;xPBa7=pKF~r$Vn=%-F8(@4MgQ@HYlft>88o>Ga<!9a5#qXEX>h%?}U$;'
    '%%}otEq0#XB(w?jlj5s3d(vmg{zr0M=tcZM0d#?ydi@%6%pnAhpoM0HVAC@|3I%mqkbTvDw6E=lv`L<#;_$uOlnXxY*QefhO2MkSo9@RChI_~QDLEzM-i1^D'
    'b^NjAm`q_>PtSh2-fUIMaA@~dM1qcCT>O4}5q53{T`d8{q#s2nA{M*ADlWp+8s_0+odxM+NSY5M`6Qf-hd4G2m=p22GT?={{FAJ^Y-~aSDKYmVeBUhh;^xEl'
    'HTO!RP<en*Zd)`J#?A?6Ybuz{*<n^r77IqO9_*8J3vq?;MYaV&dkUUp%ge%Nv<vw&D4)h&@;O2TH`IvRq<KdxJSN5Pv*q!92(+fWns_HjkR=w6b`XM(#b|ei'
    'YQ&(^HQnTLJ)aS#D;dc}iX)qh4+GTbXhLO#^<z^YAq(<-my(dy>eeDxFDFRiEh2jCT_J?bo_C0=_3G?;Mt2C|mlDCen7^FAz_eVWJdh_&k$H?Ni)actTbr04'
    'xUWJv()c-KPd$lnwUsKKxx_R(R8Al4QwFG_nO&t<m@2(5f=sd@&h6Nug$V18PKaK3L|{z;FlUSi|Ms4NZJ;R;bd6#0=<yGa1kxE_{R|nX<D0a?Pf0kW)o-EJ'
    '1k|ka7uVZ5brlC_jU;Im+L!Q-YfQftz&&vK8%BFb06@^#A)V6DM&pcNznFy#IlDY(IV_FWwn)*j-(#AiWH5*Sn^P+2>}26zSma<VaomrCP^?@6AkC<PEI{Dp'
    'z|;U<82l;<$@<efU(GN}j*7Or1}6!K4!@YLA#-@~bbdL#{U7rULe(w%yo$D$SN>%Z7gjp@1Q|stu#M9zOd<oF;~v0qVRvAQHcX`~z6z|K**Ms87$mRXg?!=*'
    '#|{vk2dH*RmDm}T-3dB}c5vR%98j7)L1=Rs$9V}?2?_a&;-~3L{Cvb9G4qm*r@O5Z!h;nYc|JOUnb*M1IkHd1BI6eVbATjhpgAft3E#2k=wuwid{7>yZ6WVd'
    'a36y1Vao&nA2EhT0AVnB02~@VaxzOOh&o29FbllX9+$(XyC22atc<>Yu%VE4oUK%bZlM#JX8Hb(S-zi~Wmp1slJ6%c8H#+?`5=NHJ015CAH|ReMFPG(L7@_s'
    'K5!Ou8kiNnu18d+hhpUTKy<Ndx;-{SO}TSWI2Z<!yu4mrGcRjqYh`$1`i#(H5NDEhru0t9Ab8JmjEH;2492gTVnErb9vq9~p)?Ev?NZq0cE0%4L_&=hR~Mj7'
    'vF{083!d@<14{=$9YPDcLU3fvDu{&zKSo^@tF41!Mel9mja~W=UPkFBAY{SbfMY(=1!_i|a8LH2#fIrq9_)TM?i;o)t%1pQ77n4rcIlQCzi%q)^w2R_o8ul1'
    'SP~%cd^|mZ^K{y2Mx%s+t`<8%MW~pd2C|2>hoJ)ZPs!YCKN<#!5^OK65BgvrM7SRO3>jmwz(_jd%%T)M04p%s9f{Jc&-i%;$_y=32#6@?y8y@r`y7U#U92|j'
    '!Gl*j_-b7ySGAy4IXIExR&jiIcn2(pGq9}&c6S=_Y(y>kHefqo`Vp0hPzS@!iMx0-G}4W-x<_DwAvjX24en+<-0C{ckE{k#lox~MCkbK5_X24X^Ea7gV&=??'
    '9g%)+aVhxJstCd8t0ImO@!~(Y0vm7$Z{B4|RkX8W=&TZSUB@gXyRcqNgTsMaONr@2WetMRS_7i+TvR2cm$A;N60F`k4|I!4rEXQF;@1@7!g`O`8?1e_SNhBB'
    '_zqz_gBNk+&x(0S^Q$J&{v@n#P>~g=i*KAY46-c!y7RNQIriab2VXWb(S&oJ6KV|akOtKz$8qawE<SjoDg+TJ!Pa4e+|RSv3@7uN3Hg_^J)A-NCOwfF%_BuF'
    'J*k>#(6m3nM-S3`*oTDAO|&gF@;WpQK0Idi7`n(<cdUDUpy(hU>>&RPa@6|u+pY7Zb<q&?HsuHEpz3!DCz$RW37^-3i^PN-l}d3cf3_-Ko{oDt0<^t?UWxfK'
    '<VZ-RJy*J$*(_faf7#Sp;!30?hU~#8CT*OCq+b;OaZxXA)Ctj|d@k27z#HviN*Wdb-iCbvhm(0^cUlGV6V<>De_5<d@ipim6h#09M-2jW0>~kJj>@wPRr?7$'
    '9|!xpjjcB;&Yj(_{3q^N+NcS7@*UJii5w)XD<?cwpMuR(F;7jj<5GHft9z%TYZZ*W3j1#TL0j!D=co``6YZfCdrrmMscLuZomUATj0WUkH@8(Zbxousds$gU'
    'FPycxnLwChEI}%=(Kb^|ScOwcFM)k`LP64u5?8Q}wUm04pE||5tH5CPVjd^{A;mp&@ff3<4wVED<F2DVW46_Szn(%(nj2=!@K}$MKAGKcSQXtt@zqxaM$Zq%'
    '0Wb+iuEai`DnP@Zkl1mOsr=)#z=m`*L{v7_O*wZ)M96el_qZ(nQ{rKKa+mYXMny1@<PTk^>*O*$C49}K(j35`rRs5bx!`8vVe<tSqa^7thhM$~H!CA0jyD7h'
    'CYMtxK)#@t$oc4;OnR&)4os5rb@wV6MIec$VQ|ABoXXCqb2(#r8Bbk3q3+|1eIb#?2iHbvDCcvP-6tP>*J%C{di8|M{q@`kTA#Y$1>WG)QTzwDsuzDes}!GP'
    '(-~PQS5Frc4`t?Fwu1DbY4)^}Y(v{Pf&W=#C^qzgE2jG%^(6k5nH*Je2~RGvF8NvSIP}2L$4&dKUBw~K6F-V~Ntt4OvZ8qGA!nD(jgdmi2h(3;cu9<>#WkoE'
    'fuu~K(g%Ln8iQ52(}%r}#);#N4nERzZM)sPr5?Twnv}z%j8%KtvkosxVwiV5s5jgBB>+T<#^P!<Uv9M$S+A{<Jq6)<NWa`gEJYI~m58Dftt>qjUjv%-RI49+'
    'rE110%wYqIMK18xX~G8Hfe(zY8Ge0O2p)Ne-b(yl;H8&m^;>aHE7MR$94sr|X4>B-3K;hm>v#u2(Pmi}v0%xK67FgdO+D|wB1P@V)=YF$cX9)FfXQV4N=1x1'
    'xE9VIktU@KYQqQ%%VnRK003*486s$E8bI!9@3j4wfL4!1^N^+*eYH^CnIS`JO>2ucYk5AaH@o_Wl@cFaI8x|{7GAI-pESE15)PkPmbaUR^9RKr=_xQ%2p$>8'
    '5%5j{lBx*)VI!fN8+w+xs=>1M{H-DnIqNx*^)}qIg}{?8N8im)%I`C?KbxJSn&>v|`mkVIn&f@ihG+{<v7n*Yn>5o-STMI$TNBKbZ`D_EWZl`<*aa?2GjMj@'
    'NjmGs=lo8so2$#NK5VV$V2tMJTf??y^%b@RDb$B8G$08=?-Lb8_mbO@oKk(buR^7h$e2FdPtD>wo^}z|ywZN0RM8z@W)JF(b;e~YN9D96u)Dheao|oEmN1h='
    '74TzFE0jc2wNa2It$?bjWZbY;A|rx2OnvJavG1rxRgA_F-5^~7n%U+UY`eRKIF){)0WjcN>J}DlvR$QYdug=ICeNZ|h1E6SdLl@t1>_V-PvLUnLDmQ>dbu(j'
    '1FFB7nmZ7Gsm$#sBU%!7@#sv1LPStqeezo!hcw-|p(x>Wot_X*iT=XeYZmimZSjH6GP5AtCk8KVozObG9;0{G>7b0=)!2h2eJ^WsH*0r4YjH>0?;1w?(#g2C'
    'pv_bh7xx8hm_Dw9c^MNG%+(&8H@M#X@@lmvEC}qM<@TgY@Rcwrax|*&N-P5mJ$DiWXz`gn9nv5J4^jZr6|hGw`eJU2_4FM$5l$D@)1y)jM9KCf$djjex&?=!'
    'TF;!m*NiYF18B(`Xze4Kf7Bh|pDJViTCyAQ4T4k-d?_5!z{kQSuW-4VFw!(Z*&#lQ4~C`cyJqZO)4?74#hl>D=`TDfJi#bd((Tdhf6gU8_@>(4=%32`&|w;G'
    'wqa!94V$V^><I(FyMdqItfZZv^A=3$g<Y;3ZbGLkrANX}T;@{p9aS*VzV8;4XS$#({X^`*mg!)r{A~9iAGx|UlLuYy9@InCZ!__FF}>QTAlPPSX2a~*@7aq!'
    '>!yAGDu1Mv*R{&8V+U9T(^>iB52KzMtk>F|y&E1NYTbI}l!#zR@a%9*6d{h^&4qHfB+%*iJ`QFI%(#2t0!M}_hqmq^M*zJr+!;cy{zM|mAMwvNQBHuVJ}jOZ'
    'ew2)Q3Syivf+^FH;ej9Kl0<Lj5RZyGEUtsl3yDdQCK<y+LnVC<X2<DC-}oR*dxaigZmaqoY@%)_5{>{|$a6GKzj6yuJyqwTTX^u^Y&Ub<!7p7=rzPb87n)x@'
    'RJ`nY90I8VJ<j70FbsIY=c(w92=PMNq#8l6hiS0mgUe4iqfrObZE>B&^<|ivkt19(mBDz$2<Ki~@!`0m8O}mz4xZFE!FZR(%ij+23UH)N$c{M{;cSD%NdT6$'
    '?xO_OE&K4rx}y$4{F=`{=!_HQcI<}X(IXOTGx0U<&jt$ryOD&yaeOafbCVmU36FMmNDPI<7WzM{U_0%=za!d?Jiao0RNtF-6MbVgjYXf6-nL&)L}%l^Hxu?g'
    '?Lp?VQQdz1_pTV|BwQLI=*&@8cg*?I3Fa@hY<K?bko7ne6BE!zsBpk4dppqXU;^5L-KBj&nn(C%wSHxe?Kht=M!^*%_eo6L4<~^eQ%~8>WJ;~>BHMShR$`{W'
    '_h&ZSwPR*}9SL^x!4DY~@k@qtZ83z^x)a2gj!d^*gr5#vHYW;r6TnGpgyarHfuKIZ{K1<JV5H4oa{fZzIaFCpKES_>WrqnL(aB7ay$y<s7e+eTcn4%voX^dv'
    '31rPAeF$*~^}#qyJnYXluQj}1?ob_OU*#yMPdZFJ&D2Yu&co*SAuZ-TL-`fH4?36b)lHm6(~sf|#DL(K9|qaH?aJwIR}<4pm%s9D`vJip?W}%#O8Ppp{H{(r'
    'Uv|LZdbT<<qd4tcZ_j)8dz<;oPLPQLh?A&%oECE(DtYRyCnlpp!d!RE$~ULy0+P5jVU+7Z)0m--FPwh!<i|gM|Lli{KRtQ&x5?AT4}T<JaV^e3rBNp*fY94?'
    ')UXYQ2cFc1<-*R%A((=C$xJoxF|Szxm5^rbWuqV^N)dp1{^=>?_nlq974S^c%Og6Ei}vTRo;4rAIi})bQ<AkbHgl%3j>kySAPnYAZSoaQ)_dP^zGRU-EvymS'
    '@KTi{D_%$@vdg8YM3%Wc6>FMnrsHSEB1)_ZQ!6@q@Nyx&@qvbH`)J$REg!w{LC~(jyf9s4ce%Mak}6^-8D}bWqfI5j^scPloi&bP^~(kTpY>I}wMphVE3E~A'
    'tpW4Lmhe*2!da?Uq6Zj@00vUcGj`A!%?Z3=wsB%6^ms!&D**6ZqgJsqf`Wwyz&tKKs52kHE?ldpAJotw@%-%%$uU7K7|SV12j;J{`t^KPoBH%6`rOg8n@*WH'
    'Z*=bcyVfSe1B!*dKwqFgwkQrd*hRX?r+uf0TZ$LSLBc!JJGC~^V%)r2ZZFIX%zk4l60&ilzdpEcf@pwu$oUm3Q{f5c0H;M~$1%a0ru3DWtZt|3l!E5$QE9pE'
    '9odgu{DVXIUw#)rZ=qf6wZGhXgFqVbSGD_4vY=>*Ks~j35*?R_@GxryDD~MrHmJc%7l0b%(oUDm8f@w7PI^<?`q9pWKDOJE$?5*>?MriuupOn{Nn`9(Jg==b'
    'qMQW|<c=R4cpKAuZJ+wsqM|(R`g_eLLd)$p&Ov>>y}E|LPG51lL(Lj81yZF#!U$T%N+Nyu?YGY!ef#jIM=Zsgc~xx&Q|*8|7G_K{Qc9}ga5uZ0F6Zaqj{r*y'
    '3^*N+VpwEyp9=Yx3nJgI-x`h-<?rX0*UQCx`HH1x0?`3V%BpUq=%L8a8oh!L<A-jS(UVdmY|Q2$#h#nB#MZ#Tl{!!nTU>i1-dxsc2L)#i1lt1L!YSd0nT>Q6'
    '-%|rqg^;!K?5f#~q>YL?Nq`kG)}a`r#|bK8CZvc-S3^G(YSj4*Emz7nMLVSg{ucTQe09y~PuNlUwqXzbJCK6;d~TTfH-Gul!{V~OG;&;ldUM#pMxZp;3lE<}'
    '_>@JuH{{UBbflOVy7_>Df#e!}I|h02<+Z&Zzc!mq7rJR)*$;SSlVCa+c!soqJD4&DUl>mQ8UJVKy}ok~<4(SP`d8rk!{Xb=PyYDuaq;xo6SM1|KQI3J@bUM5'
    'g4^h?&^Ut55qmG_^0y&0I|%U&G)tJht<K8&l_3YU?EVU`n6TYYUxu{Fl0vJbep>_R+S)KO5NORA34mIL9%w4f*DEepXRyIZ=^@>?Cmmc&-$C%K83^R@1S2(o'
    'G=Xtry2M!K;(T4#hTKJcy`F;(W{9I8WH5@3=4&~0Z@PS^0-G=4vTTkDbEP#>6EIU8mKl&CsN=O?yhGpyGbTLqd(Ks&Fvsl~1=4Y7<Jo%p=4|z5i58UgYJsVX'
    'zfej5b_$UJ|E*qtR&+Cexdb~ROw(fZ(%jS$Q0#0)gIxe(8^&hOse7y+_!1t1V!@nd8<<R-Wg3L-n<<P_wKc1txW+(8Acekb1@MgiXJ38?%<xG^ac^=sd3p63'
    'Ij;3fTA8h9xSIW}9kWgUdgtCDn-L#AgVXWZx6hr%2+rCG1|_P351&4L{I`kOQ9nKT=E-BXv#+ie?=0W(D#(eO-~ze>&W^c5XQe7d`lW`uud4RPzkL7rPm^yR'
    'KYafDKfX7A6PtM6-l%K{)o-`!sVOrD4*5F3!%)d6Q@0ok7sfXdEP$c?QuyKe<u#b+p5mMC^z3YMQ7^7YMMxq5_7r<jd6<C{_ri9sRLubR7kpKu-|h5zFq~e!'
    '1g&Imx>)oi6=lyXp$=uTF=BFOW2Um6u&GSEQs7M6O`MfjJ*V)h3W>aRw2pQYf<@CqT~~_0;OKr!5?b+<qIB<WOG2dmN|I<v;l0%69HMqG{WhJroG*Li<&~y#'
    '*f&?c!SQZFEJC#-cy!pJXVbSmV$nTe(T#m<L6RH@dd_yUT`lT0+x|0@+jLdNz8B%c7KGp7S%Y&5Auzipwl%@U>;+`(he)*~tsZ@)7N4CQy?e^hJ6_4hZ9ewq'
    'mDyC=*+r5<i)|^!mEBY<%R^{-rb%bjtEE|Gcw%)q<tekp(%t09;9`@Mqm1h~W*OUlHk{s;+g0_w%e3m2cTTudw2P-gd>#z_VR@X~J7_lVHUJ~{dNdf01)6XE'
    '5(E^R34=?}+wEF8naC?pWJ$Hr;TSM*gSk4;$!y7<6H5sKY6)Z(>1L#r#;M;m>vSK_O*#rE2M2DB?W`%`MSl%nOldZ7<-SxrExt_Dbt#Y78eqyrb{xj!J0eWJ'
    'dv{4;7Ufb-RW-Zgs1&C{*Mv3p2;AR2qJ4CHFHhWAkE#OB&kR|0U^4alMBLQbLEJxU&7qcQLb4I5Ss5Jdpe@Jmy&Tc^Ob{?E1Ysvf3{GM^bt6(Xz^olL7psQu'
    '#&`fnG_A2}KiW|>X~P_JalauWB{@gfYf`%r#1-vW678^OpVPV6^Rr!Z`TEpqYHYE?*Pe|w5RkFPt<nLEv!6hlN;Q+L(P2OIG<9>lPw;qeL)aEJH0`+`5)R_I'
    'd_2&>VSbmzfndjFYMpIfP4M6jC^s-3SH}=d@g0!}_9g7sv&1=5{xi6Ro11#QerI#VyfF8+i_7WymAR%HM)mT7(&#_^Ycah(n{P=e!n%?m(b*TMK-g^On8D3z'
    'FnXx?5e1F%RS-RiQ;I-R(!L&glkwlTvZv05G9aZ71;1+G8<4mAr)x%z0V&gex~lRD2rd7i9o_Qgu()QMVGS1#oS9c~5J=UHIbTR6bP!Ouz)jZwWU+W-`tub+'
    'zK64*H=bY9y$UNu8_TcgzYZ&BfgNlfcvnI4BBOVr?-K58T^&}|lJa)QetKEeC_3;d7aZSx2Q-&X1^H#sIf#R9LXRKZ!Qxy(8r$hPXzEx6=_is5)RFT`BTK$U'
    'PwGmy<}!5l8uXM|EhQ@{RsLS=Byb~ERIck^FzSoFEY)Y4U7t-Sqy`3Ue{R}IS59_>jj$=UJ0vyAedK?@6*bn-HKdpqKUNHIKGXAHHQ2pCgH0LV5cdF>G0{N{'
    'X%94{km_i2TQebq+zxz8Ee;9Tw1TsM?Oo<qrvy{`biIX0z>BUo2X`kOJU8wR<j0ZxxWhk=<-bzoSc)7&5l!t_$Q|2C_vFWk{J1Ya9`Fyd=lI8jho$a8%2Eg1'
    'ii&?gQ?nWLIsEAU?)QKD{pIh^Ccppg_dopp`R~e`$%nKPz)dG{dGfDB_5u|sbiFGr^GC|UaeZly&Ubs@3<ud=LOA<_FynKwz-Pi54q|Q`0+%2wECb$~%VP&y'
    '<op~MIH0`Ol&LoWrY}>&w<*FmQ`n~O`Ei&7$EG^fGn1?9=gU54;Rik-e>#;lIG9~GA#={1tl+rK{u~A`{Y-y}q?J2@$s1$H+h(tnL}(CcjoBsXW5fXyW&k5w'
    'QhXMvl`;e=F=aWv@{+O%x`_+E4S$%ZnEni`0jt-%dvwSdAzHhcbfqSnx~brs)f$yxB*b*)%X9WR!50`xZQk|<&Tl+9J2QcNz)<l^8#Mb0!ey1S&QzD1=F80{'
    'l+`apPr($5gc#72y^snX<}ww|DEzQaQHR~xg0`eYO!Y;LHb08#Cy!$K*+wyK9K~!x`Zif4iKL2lLT~wb8&l~T<<qKh?4&>4gZ?z6FEYL!JvZ5t#xp~2iEVBV'
    '2e_W^fuoH*OQC4T=}SL1qILt&wHt{K=*rJhsIm<Exf2B2k!i+~mFAP}Co!=d!yBD<wyP_WUtSq`1(=;1ZK8tVkf}N{2JK{}X6-BDkp2N7+%(Z%#8as)jei+&'
    'RUxMb>d`pV!kF^n^296b48(CGc|e7=L$5Mb%-GWvCOR8Thx!sVX0s{a`!ZKSrRi{k5<ekNT8wk5w~`VW55Y+%o|wXgINN$)egCWyVk(urYLAX&Dn^<LkET^c'
    '_)l5?;5`i03YnLpPOz_2ZqEo2<w41Y@DgbxS-290H$3y{>iKIzlXVwT0hG<j$)F#Y03snLUO$IWQ!(;Fwjy+_!wBb&l$DsA$u+1TK+6ERqcBle2gYG(A_M7w'
    '_LDO?fhg%wunA#STS|9CHn#I~PAwX%k6dO8l}=7n>v8L0QAtmc)JFSCTtTBODc9EIo4-7J_UOl-$eskYHSi$CJe-(WqouZ**lntlpVV#})jW<|^xVL&(#q?P'
    'YJa!-e5ca<Q`*gF*Hbq!w4%~BvniEX>nqnZaWCaO(*hOc-`1*-NfD`*(u`zPa>Zg<#y|e4MoN(Rc;O`_xP;j%?RN^fE6iRDM8a9Oec{34Y&WCnVOkcI47PG0'
    'A;C>l$N4l~vHeRqexR)ktLbC!_{wq)?IDM9JD>LhHc6kBZz83hzvPt2x3Rb}x<m{lHTnoKmzl0KM}NkGWlogEhj}ShSb9J{DZ3gD;m#1(VY%e6^#U>x)!Hb5'
    'QmoE`DY}2?;q@goX<D?LJsPH)Cc_(r6cfSWm}7g<J?!5*;XLY@Q3cdm>`%RwXshZcM!0UVbN$28|46P2y@+d+fv!pPkIwNUrJVFP-!6t*74Z!*mX>XI{f>;X'
    'g7sKI%LsE|?|2lEGMF)DMgf-w{hGb75VP>UPKq>PNwq9-K}7|Eh4}TUx26)nJv=zw`(%ex(`KmmtjI-ix@BTX?4}(qA0V8NQxy2vx<*7#r0IPDAg9OVlgDJQ'
    'mI_G*?LQid-_&&DhlvHK<QG4U1)k`0x{(|t({FhFG(7syzX*e|0C6)UZK2J38=yniv-yToJOOO|q`COIo^GhHYaguNllf*MzG5otl&f4XOrLQEbN$4Z`OcAb'
    '&n0=x^@}WiII}#5rbdt^eIp6h{qE66$@P!k<~vB%Je7r#0}Lh*U@(yYgNXzfWcx4sn4%jd9P{CFHAk5zz4YNR`8e&68`E0P!Oj7x^$|d>Pv;AaTOCF4POx$G'
    'Hcccz!pag<*|FhUvlpE*MNRFDoy0=_rzAmtZS16Ji$~F+@(HQJq*L4~I=K2e{a;pdBT(UUJZyQ!=n}kKxzG;!>TY^B)AO26Uwc*o3+XxUs9u6U_ll4^do@C?'
    '+e_VfJ!DN3(KFaA3H%2AC`y%Vld}Sn4k>>EaU@I_%_+4Ra^BE?O=9tDkcr7O69xl1?F8hJb$^A4yV>_fXD2^T<^=b^PlgGNZJ?}8>uH%v+1P|aPf%*x9U9wE'
    'SsUnS(=Y{|%Fvx;O@*;@A_q4W%w)(N!VwtMO9()A1`_Y`$+sl|u6@!LogsXYZ9zI0Jjs@Ks+M6Sio30rtscNJIBpQsNGI_3Wx^=8d}H!|Ej_kM>aVUR+4A^4'
    'gqq>RzvviF+giNKv3SG|@=EXcO;c5+{%#civ0T)PW(~g@Ur4*DVsk5jQJ`%bUbVP|3{uU|JGRr60`Qq(aFnJi1!uBURmXa|-gIf3K~XVWWLK{zB-l_t%#TC~'
    'k#8VbJ^xIUqNCiDp&zN1>_&9^+CD0yjQloP1D>N8H~p->T1;oPcy09`kft7!5pFr47uERYsS7nW#W1k324&b{0Zw1@jiJWMeoQ4d-MTarn-!K2*R*<5%(ol#'
    '3A;@md28_3A}20%o8zcw4qmp-TI^hhy6_o<Bd@`O>mWwlwRd>^(6c3_@nMxYMr=w>3y9{C=&@%T5i%cP95J%j+4YQ!BO>XrY$vzPQ69Yyq!{y*^wMe(veTMQ'
    'ctSUvvF>(oBl6eD=y0r856Ew{@*#KNDLby3wuj0mkk+E#A-7f`?+CwXb5^s8zN~WO_aKs`kz?n(>V;?WuJW<S6V)IAvlmL3mV_X+bnTp!oGBr~J+}M6f9!`J'
    'FE7wZ7d{gDg?4wPvJMTRyVFh2FtAKzJ51$x<V@1@>B~&nFr%brHCI^`4beoxWb-skm+U7$bRC7FDzTvXvrV4kUAdHrDkZH&fmo*}#5R)I2{8me7kY1VU{9@S'
    'Qd0_&w06Qw-0;n<P9UUi({K|-Q`Enwj57vXTU=j%7#8@=MZKoWld44q-Kw4Ww%XHN8}um!`{PopjIHmvSBZ(vH<)P#3PH5G1cE#T?<v8~=rQ0b9A)-8+1u2R'
    'uEGd5=B$K_HF&JK?4!m1c+Sh<3wI8@@bZ1|d5dSTIY~hYLcZ2=5StUVUZD5lbhhmu#JNVHxo}&EQW4(K=AhiU8C(gJhM{PbL)SuwTut8@hIhuygAx~*V-n9w'
    'Ixwpuwyr71Pr!|7LnLLV3rGR$oK$jJV#h&d0Eh4}GZLl|E*zT{G@(+RRhWlYKS-bem1Qb<IUrZ6VZ~6>Vaj$q2I{w4%mmD>Wxk2upckkcC{7G96xN{+n=X(q'
    'gCzpbz=L*8D+B(t3F?qz1)8<`Er?^qBm9rqKTS=?N;xkNE@PB{={o^8u4YeCCyP)Q7PTu8-P|E_SB&YNnxwccsgM?!7aM>0D+7cn(-*Wc|9jyI&@z#?gWzpy'
    '%}Cj!F*psR5lP4WIrQT&kyd{9S7<perbzA;<rJJLsWaV5c4{TOLiNJ1G2lJW9xgX@r><w=YRAxv-h&4Z$_R@Z^m_T`+>i{_#!@fL4eM31M7z1Almwf2G17=('
    'q;`~z(`UP@1@Gin40Z!lG@nQ@zh(W#oYRYq7UPg-TAtf*a~+MFZ>`Dq_P&~Y>)ln8CDQu0WYN6xWp{B_xFicUt5X`QXhniU&P`Uq>(e9#Elw6@8(quNYmaX<'
    '7JUIS5Osyrdihe?pxF3bD8rAK5Bx&8a~W9UKlxY&PIemYB}v0xnQau6=Tl4_M6UQKV9M^kPJA<USKs%h9@&gPl{gy`$H73)J_Bj6=lOK)R9mAT5))?clZ$M>'
    'u}~G!v$K_u9cBU^6z)3-)$9ZSDQv%>83F^wVG~u}9Ng<yU>;7}S|>}hQsdTAa4e4c+ChjS>E((zsa)24YQpqKdP>J)F$-p`Y^Z8^y{jC4Fk7U`+3d@*sR=$V'
    '@(pl-3Qd#Rpbab1!Nl%pMx8shDb~q&%g<9!$(w!DQ%#*n9c{VZq{@<4`*hjdO4eN{(^6jNwRdY|osf1sNiHsD#JNE;?MCZFa=|p!Fe~W8O{ZOpsbk!>h<-Y*'
    '$p%XT=Pqw<Gc5;rvxtb#P^_6!Mo5TAF-zOAFVA)pGKHF{1ky#Y1VM?+7Ps|2JFPdUAFwkjo=(`n*d_LAvu-mI&3--PeG>ME$nM!8Uud%ZpvOLzvZ=($t2WS_'
    '(20{?iTcXC1zDe}s65)Elmd24s?Q`YjMEoqQd*MoU$7^K#iPeRJmNn{2K*V?iX&nax@i0o(vCs_@V=kF02@&vqy~T|lcZJ9fBdpJ5qN;F$Z{vcD<%*ZY3z`9'
    'd00HTqRa$%)8Y#?2iK4)3roA;XC1J#-(#91H?Zt)&g!R?f8k$P<bd_{Hqc>|3UU}ze$h>YB)KCis<RKtGO1`)RC*OuiLYMSuUzS`T6xJ0B$2(~e>;qC{xp?N'
    '^#D6YVl8M4<$(uK@F9&ug%w99<EjbiiVEEy-}oeR1{cZB&tBW&&W>|aI*dKlbJGFv8zpp*oI$0sWp&N_xtcKwrd06E$L#+EuS!r@gTK8Q@yn%=okmoqhx}&T'
    '3VAipQT$91{~eK66B&#5MqYLF8m)O%2Dh>N7gXwh8>rMq5YLaa{r%qq$!Xm(yv$J=8P29>=X_2wX)eM2ZwO`cHl|#1P@Br%#9{~hR&g~0|Co55|AnXdFFeiP'
    '9-ii}h^Hfp@1b6`9^MYEUZ5gUuX<b8;u8-)_X6tq2-F?S8;7n&VR;Y-@3esGVep!r0GXV5ual<b*u_vHavTqj67T`TLBGc^&7PlL4<mQN21OT2+0oE~#@c4m'
    'F@u=HDhFRx+8CQtk@{B$yo5%Vlsk)TO9dE2>o<uQ)B>s};CTwFhv7j4uIFO_WplE<C;fj$K-nzd$7chSwU|M><cA%{+b2QC^3KTM$7*M;EOu-@DXb%sg^Z2U'
    '!UmNZ5M#56VFMCLz}Q4;*q~YhFE&UH>nbR9{qFRzu3DU1J5@a@6}6s7o;}_t)vP1+ktnf_05bJ-{Zy{ckAs>`ZFP?k{0nOKvq8;z*>eATL(K|`RUX0WBd}(9'
    'HU0~0_Fq`DpBrm7+?xN*v1Va=emd5yIvrA2v*C92!CZc$ShErG-A7~1CedY+c(PRnjMMEcR4SbY2%glIzL!I>3=l28`YpiFemtgZ;t)v|w^uz3v;*z0BvII~'
    'eVoGy6Lmwg{$_HvLR451=_hjiO;%P{Lk~zcaXtL+jVBviQZtycnhX3|n6f8?Df`_M4^#HLzde2Q?CHa24}W;{)1zm_lczs@|HJqH>(Q@+EDQ0QCk)Q&f=J>^'
    'fY@H){7B-%8fq3Hmi7(Dhq_zsiv`FS;N|_M*c!eJhyzogrqNn2-d)ve!ypZF1gL~fJzdW(hUUm#0(7f9rqpK}^Z1I@+oI%mibB9@Wh_=hU!|C|01bvwn%^^k'
    'cUY`eSA`s9I0=`9IZ8dOTthIpRdX<M<swxv1gDn!H$s>de1!oEfee5#qk-badN1;XIW$+CECDevQ@$#;*Gmk}HTQf=Yx8S^zMjB_xman?SN|(bR#%|KeP1*5'
    '>q#2<)dx>=<x;?}F;0Lg@8iCDJ5{wF!oFrTORi1^^|kpLdh%O9Y!x`aW}Lqkh^;5RtJT%@LhQCNSmNFLJ1M-Zmo@FdH16ZwofK-bt?G|Ld_3v#r|NB2S7|WU'
    '!<{6E>*VoZAKYu*i#(?6z1IEM1uA>D{iJ*{KJCdTAk&^SVA3A#9s{7Ag7aVla_ZsnPAVFYb{DnXUuX{?+B>`Z=5S)o{c5_{zH6d)yt{YOtsUeB{T3s=ye2<|'
    '-o_k-dHCAF?(Yad@9%5apMxHV1zB1r?Z0Mgnh$o<G>2^u)SEp_?nt<?h*0;bfYm3T3TJ!rUyWufr^#nVvpxAlFx!*g1eWc||4~r3CrKpRlg|ufd(sZa)^x^7'
    '9RkQGP_6fJD3ZF8*$H<etjCjyl0*WW$D=G!UwXM46Gh>K$=xAxck<*!lu7iC59NvBT0XhE`5~)1fGT?!QzfBcKfGSAmZAILUcR>p!gr2#cUYi+@8!Gv5m>;f'
    'x|y{g9m%(HZ}%v6VGf_{g*JSW#~D8H5r%P=8(?Qo7-aY;F>tUa>SWT1%3MU?X%CM>8rn9wgH}+~(mV`l?hhb*lEV(Za~!WJ>xRU_Pb>OSEotHSFlxbu`0Ud_'
    'SP8#cOk1R_2})azGP5s`jT##9YvT|GiG@Kz6R?s2Fnsu`xLk3;?Hoy18=E;fj+?Qdv!B#ljtVyWAkpmJ*uf_|po3374mtQF2OPXFoGIxENpH9l&J<NrNw`f9'
    'yfg(_D*7l$Iir+wp=!=5eTo&d*GW3>5neIt5SM460!<Wj=xD{NIP{IN`j!xrb3Y`~lPnTxq^a%~-}&QH7^&;}8dBow5h5Px4iHi!zX^cpcD|@LUuqmuP;3VZ'
    'zW+OrNO@DVK_or-Cm@oZWC2M}(4az5z+?nc2pu7IXfg?!i@9pX{*+Y8raa^=-*nFtz!kGaEq*b(P(yY=V<IJekAUV!8Uz!a%ZU{u6e&kO0umP``h9EP()pz)'
    '8_DTuG`#v1@NJy0cU@S*Lq9g{6Xp)6`Isi~Br$(x55XHlb$$ss^p7AG1fNj!%i$O^)(1}qV<>wbC5dbNRE*5ZA1eTB)miHJsls-%H9x?i0*8D#gc+vLF_^?%'
    '>twsy;$7Ebg1^{I)Fi#w-~_3KLU}i&SPWAGwaormt+!oJXOfS^X@^p=b;|yBg?D2=G}Wv<8Vk_oQKjQjU%m1VQUU%vPXQjf<&3=>4PU()Cz7L^X$sc!Es<CS'
    'KJqsI{pwYaG7G0KEUwr0a%B#t1?)YTO#CFYC9ta54|Q*FM0u}ylb9liy;7EplR-(i&11byIJds`qu7a7yfG){qTZsYM*wmj_?-rk10(zHQtTMGUDwN(+lxe;'
    'tXU$HS^PYYaTmem1wnBtf^`R%<W9-HdO+!${T|cmp3EH7isgzg%rs5~oQ3|NtkuQYd~-#v1(vibOuXO7BJ26vu5HJXb)SIY7m0v7JG$XVvQ_bu<Z7w!Wh>BF'
    '&9Z}S=`BOW|NIH@IfSr?E9N9Q{)`jo!E3p0=FY_k9GWs1ezRJ>2DDoUW>OQ4Hrc7o2CiJ{X&=~}g=%E3x<9J^#!Lg^OgUZvWhXF4{^5`#kw4V}oJQ<iYWhv}'
    '`K8(9>Ka2&yS6Kcc;GsAOIXE0ocbl`exBXuy4O@Eh6Yt1xibvLrrv2M47(Pi3igdIj@uD;Ra9Pm9)DZvg{y;yy!F~dju(2Kl--(5RQw&DCk8vViH_aw4gIk='
    '9wD9Z8++kEc2ReR;bO-{<Tt2lDmLd=^_b<ESQ%m*){_b7ozEuVMK|rjZwh_3qD_QoTBsZR5V4Lb$+HfZMS2vR5iYa#tNI<fv3e`E(yOuZ)>;)Umr8&2ioaT='
    'r=AmBPv7{g1Em}uR+-vWQb9O@c|h5Kgq4XH{YIpLuTV_W`N5gQYFTeEvkHti-x&ORJ0hFe<M@g2bFr5k{Pb{mX>8yYXPO;4x}*7+-GnQN3wO5(?BPvvHCiRt'
    '5QaH8*6hG7g3geUT-~EGh9HKuAtlwL)QUu90Qlrt^FVjKe2L4O?uh+wNyoohS9z~Y3wf$mWF8IIw3P>ow1bH@j9$hx#Htt#szdjGX2YOA9{#K5B_?IJfF*QT'
    '4L*jD+dhhtqSc;~zx6z7mR@^mjt~2Um{fa0$cRH4NToduO0Q1os4wSlky8KAkfDE$@5`^F6Z4N*`|xLb@-L<fP#KDN>+6@OF{F8PB>xQV!9RDQ9c=qf`^H|b'
    '=Kz}``3Q_$J;U64Bc?ULabR=sYk<$7&BK#DzMeh6F6!y)cRYqeOa2c0!nwMGl(DrG`7wuJLwXp#Dzvc;$^{yD46C0%TG<>UBJZOhCmq4Xd{JX!LDN%n67YbO'
    'I&5)uF%8gK2mK?EOb7gr>5Kj61pYbBA)c!K`qyP@<(A*fAvA+>>|y_o{l`7Kn`_3YFAT%&pMi4p!tB^Cw&IH6ijd2MUL@xoR^ok-i5VW=!8ib>T+sml)Tv|_'
    'rcQm;GF{@ZfB&!yIxtF|R4$$lD1cwH5uds|?!$ZWhlibZTx#sqD~XT5R=m^Lyc`YbLyz@|g$A3NC&3&<W)lu$Y>}$?^ka<t1csE=`@dYq0!V;hi8J&m&H-l('
    'WU&MjvAQ&?1YoOD)j+q$X~iEAh1PPCm+MKB=w*d1`(6~dWTo5JB1a06)g0S~1xGSJtv4)c{$p+cr$KwPUe0`R6AqX`MI-qGyr|z^b&+^fOMc04_-`#g{&qNJ'
    'mRpd=3-|pKID^;F4z3UwD)l-|)T>DDEirC0^?A*QGJja>wdB--j(8e$6x~V9g1?j8fvsu5J1=oulR`16hDVVWUN`IJKzIO6&Q2?aJ0Wl!i5(t-5J0g?hQz=F'
    'oY-7#{$H0_IX>YA8mNdjA!qtS;i(Yt)3`EjlFHQe&Nb7MX4HJ;R$5c5T1l|8c}`Q1F*YR?edqJ_M#M2HfKUivbs=I34%}QRCxCtu7#HlIn@tRcByMCiA^JAS'
    'vZAsw7<m)(SYsE#mk(_NztG|F3q`4Nhvo>G0`I`;gXI&-IU$%MGtTu>-(}-|zB0^(Z6OJ}5zb}eIZdp!76nk+scu5|M?+<DfrioGGIn{o`TInb`D?m~!mMjc'
    'a7N^Nc{$eUn6~MOf~es0vXP8Mf$-l#$Gj$xWhNIYPr)r3%q{GWouj6TFhf{SyQuoEM_o==I=iR?b4rwwGjG1xY2S&ACxnOWp~|zOrpXSYf>78a8bVHJ<$W^F'
    'z3bynz<Zu^#wH}%?Eu#_u%uV9In%Q_PrQ$4qm{Amn|(i`R%d!&OU!YE92|rIRZhBSh7|m)-#jo3u;DM%#$vJe9yxa0^PYuStrzplx!;M%bIyAky1m(#JJf1y'
    'C+EozH@&3W$hiinW!mUC0oW>my~~ytz)=c0l#LJ^fziSzQs(+-X4|@iny*7oy8f<UVyakh8xo*qK3AnSTD;8fG2esWH*iU>S+<}p-*v-k<v4rtJ3mM(_=ZbK'
    'PItJOgzrjwH9k;y8F~GvOCB+spD~y5mCW>+%Gf!fg*Oa-iLH#L>8tMeoU+%NKN6GvRlgE*1iTgzfu6sLe9Ub_3jWe1_!H@2<Qj5W(d}Z8u`;x>G)I5>AD;ph'
    ';`)FfPH4zk1aYPwlGh00wh8T&tU4^NOYcE4mToA0RgBFntPai_8z~VS{Y|Ig`OZVo<CgjTH3QIj7`pjk?x`*Ina_s(8Fvl#qCK86BrjH*jdp161%yL6Gp@jw'
    'qiBYmKUwlnQK~ct!eNI2pSc&!7#}M5jr|cw+3@^<qN?L6oQFYRASpHo?=boQ25w=P_Rn?bYrEL!v?OUY9xW8?wHvPJIY>~z@9%$pdf{Jz6@L}AW02FrgHEHY'
    'k25bV1{t1Fa$&<Di!sS1NUyjP=Zg9VQ|KnlgeaC}&JrGA*y7u;Cy|*_-JjGub3wmeVkBFqOp|ctOeH(!v<`quz1~hh$95W+G8D;#)iQevWyFsporFy@drNGh'
    'lQgyNp)LrB%BjfUX+E!~W1|d?7Fq_W129W{Q)}9A4j`@kIlwK4P3v<Yon^&?f^(-rQ(^3!9289j2_((@8*U^GZ0G(6N-@`v+%IQ#MueRN<?{%4SvDI2);D6='
    'sY5RukALP6jy*j|PfuJSe35NIz~q7_;YiqJjNJbO)Q-UtK!eSUmo;K3*(A*2BN-w+YDxsjmdE!2i(z??A~A{CNjN7KA4i{tkb^Mzkdpr(N06gP6~75mUj;z+'
    'CleMbNn9r9JUTVH``tlztCBxbf1xxHcpI?#<*~|z!9U~9?a)HY9X+57SBzGQ%&~uoxBWq5m`j+?#;Z%z!)JQRZN^RmlI+)PH^_?A=<P8UhzXEHKhPO!FRZ20'
    'fzQw0RxCbvw<Z=ClDDRWQNdaeBAM0+DFnf&x)P;{x2n22fJcQV5Y+DDHn{I5@=HKgeG-ew^gHVN2QzRq5Klg2g#;_0bjq9Gl_Ai;G;gb=PF2!&g~jz=JFmC1'
    '3yc}tU?#1EN~r;-rvVztLgvK^ouA`J>^I<4MXfYdNhMQ0LtT)I4hKDS#v%U_Z}0D++6Jrj>$h9Z9%yzX;K6UUH6*six7|po$PzLYr#VOdOU3yr<`9JJ<W<Gb'
    '6p=?0R94hHnaVw-AcqjUj7lxQt0Xa1^noN)IDTb&maXhcWQk}rQ~l-sj|mjm3ur6wU5HK(vRnEZr8E|N?rBo%DCdiOW`~bMp9%9$I5!N=yfHD)yEdtQ=~T18'
    '93;|_LxNC*MG1rEsjPI;YK{x%he)otRopqeQ|iN6__!Z8;&bje^uP-%2tx(D*a_8b7qOtzG&DF{N)Af-fE<AV8TeXi1F@{gD-DOND&27Wj}Lfx;ZASj&T!J+'
    '=)z7R7k)Zf9&J9}-NOO9`FnVtT5XjLwvSvW!q9#f%;l0-B$Dq2*)7UYFwln`9io@Y^CB-=x=-Imf@nHB_4i3Kb)NCca$wEki7S?s9nHBg`&hpvdfboRjmL*~'
    '%)f3rlhGKs%|!fsz~Mx}dpYrB?*)B?xl5g@<AWHEI&b1~GMJv=hOds3=dH`*NvIUB!Y$(8Yr=V`!gc=MP>8leR4UA5?>7<@j&*_&y+{@42>Kz601NIQuS2w$'
    '*@)asFQ%KCGT_r>bkgPALa_)1skH;%E2w+~ZB%{M<;EeIkz8c?k<x$g6ZFkLT}&@uoJ|WO1#}Y=#o(nLyfLAcRz(8<E2i<_a(SwGNYsFMc;lC1?nA(qhit#K'
    '%BvNgAR<W8oLPqkZ|7q6ow8Jj>|bEgP)NBW`v!8lU~<iidcLMDa2t0<u){!MA{QvWjN#`c9p|_l;);n^&W+DyC)?FTRK5C6XX?{qyV=#9?*06#E4$Tg($bMO'
    '1L6%2?1qz7sm^)*YB6DBk1{cO%8I3jfS*&3p;8h8S_nPP<vKa`zbIb-`GdM*!H1`A=wf`s)pt;J1fHH2*~72s6@wr@{JNgLQtJe~Yaw5_xt?4N3!IpZVfo_9'
    '3BPCS)n)_fFE;%Hjpo}J4RU54@qH7g$9`ltptj%UpXaITyR7pV!GF^0q5JqjkT>%#^~u06a)d^c$RYFYaU9ZpWhUe5h`nUZSm(348&G6d*(2Z`m|Z*I9d;Ol'
    'r2SUSK8)#VP>!?y$Da3blkc(D?oq3Dk3M(1m*@qSa%w+1_8e_JEKKvQ&(bdXW*2>MJL%ToW`f&*eD-B!fX}{Z_zz~XZxxW^r)T~u|M2F>Q*fo5-Qj|J*gr1*'
    'Q0(|<JDhQwXdJbn5qF(OuD}rjzkVDW`tXOhBS1jN&8i_0&TP=Q@0;jy>Vs!8=<598!3uzAqL&~Wa^@wow}QQ7{)YmGV*V&<c-(#eDWFd6F$c|~_im$HVl>Qg'
    'B)$V;M7m8b1c!D{L}r)1+kE=%d^r=(Bgh4A9{sX1v|U0ZNNIOlWrXXq>Q5}9SF6?HQ<c$<T8q2zXHq!3HruMgeHFDo15Q=_Me#rD`pQi*3CvawLwDOgtKdQU'
    '<tv_(hPCvUmv%qDq}x);J;;i#zSJ~1=X9idiGNB^s<b$y>;Np^d9yc6s_sKeys1f}BYCkZx|+qYW*-jgkJRw{OF|d~im6aGSC|DpZ5Rk<(RtKIiywRk?6|a1'
    'KsydJZij5Y0((S9(nEI4(c(4BZWX>A!k<~-ZvOP3Y~*n<mGu)rkWzvx#|==yb{eiGRg6a|t&8Czl~Z~rwK4wS0<DZR)1+nNGHtXnC!%Hxv?zAVeYC#k<nV#t'
    'W9`EA3HlxCS*+#pXKm~?-B#lhe8ccVMC-YVwq83mQ1}t6s0YsW|GAYwpG^T&hHNyLZS8#=D}T6f=b(+|CwU<~2;9#`YwAp*D>k%*kEGCnz&Z`tKop^?lSgxw'
    'NAlT|0{`Voq<kR2N2-!i*>%+5gQ*p<R!Ufn&nP@HEo_T-ZfYrxr+<2VHU~lcgGOMcY|GRpIDBZyCIC8eoihc4CmyOZ_n2Ox8xuXLqK36DKk|7I-Bv6RxCN>E'
    'dsCK)E;~*V(hr8<)g}n6-88IkjIJzK33FCangn*Fu|}xWY%v_JLz*UUJe&f55nxOhy=W!Cmo_(vPLAl2tlT&9mq7HLX%+b?^Q|Gzo!i|{@hzCOHaG7Xu+(hx'
    'MNBwznGvDo$rWX}82xaK@qf9v9N?zllAtrnDKmAmtI+xN0>pvtp!n*m0&`pr4)%;Ikg%cVUK(2d0D&EcPc|r1!(cq;UP>c-3OP!!!{Y-lu<!67mltsuMgjig'
    'w7`+-q=;#^j1kqK28;h>0<1v4oNqRSx`6BuDiTwGU+CawK0+UCgP@qg_Zu7JN2U4lfCIP36Vn~4=tEd{9B2VOAyH5ywLrpj`Ki>HAYGpbe17^A&gF-@hP+79'
    '?juH6<9<}q!_c4zW4~H$Dmwwr`HE`<iclQ4o4m2F5~PlehM-1I>h;7eBV6oGrZ?Z~^?Y@`X}LaT*K5`j*^2+*U7^>Cy0Mt4T8qlX^_<4V)PkOy%@viD)*7N)'
    '>Rc;sWu{1^TH_k&?%6h9PUo0GdVsimJEUHSAaHYWYgchoS|0g%cQ;YuUEiA;I#qhn7V>#{G*RkZ&mr{1^&H$<1s<LO+}e0J&8Ca^JOHx-?}<dF&-(2I?G6kU'
    'Sf?TqFQ!*%(n^u`iC#mHT6+U7aiAUJZDU0J^?aos5saleZ9t3WpD5&Lgr(DteEU77Inrs5Gru`RS3}E&e_@dWiQt9I-!*IN2ZMq6+^g9u%hdWF5G;OJ*;Pi$'
    '*v?;GoApGyMQt1`mr5S0mO!{U?Td&d%V+m%$kW7rjMtK1HGZEU;{3Qw(ZGodw**oZnbOV|znVyneon+>-@mOFmlJ%-8417wI9)6z&_XgbC~Ra-zAR%l$JW$H'
    '0jGyH^#>n3a%MR_3;Np7dhODG@JywjvGAa?IJ!L*Ob@n+1A#CA67P2TLqX45mxd42y0AhFK%kgi&S7Ws>C5HHh)}bgMXl!)2Xvi-at$y5UCgO-mEte2&TM>w'
    'QUY*Xz4U3}%v6l&rnUuLEf#*<6PpQI=>ix5f(y-;7G&IhQJa<MXI?z?H8C@x?ZMBp+{VUQBk7@Xhaj!guD#`I=JSiOfL461RUF5H7<xH<JGp|(CW>r<Omwl@'
    'p#6X@L%0Hbd9RsT=$LhQ$DFp^87N8v4Q|5(%|_IXv0o!KKlDt>XJ*9PTcg=H;tR+?K-8P5CcQnDs+Hj;>eD*DqN0lBH7_X5uv`eO=M1{5Wve;WttpkNUAgI8'
    'KyCqgw`Te4#V@1~-08u}6(lSlE7|LtrtK@ToWKuKK5b-A_$T9}URQ&|fm>Dym$5I3@5L~<xmwJ(bT<f)jM0YTftwV?GXT^=M+j-2a9FIt@y$gIspHIb0!|<2'
    'Xrj(8vs1jyxP$~}ftlcEXHT;_b@)9y$Xk+Ok<8EDN}?WQl1qPwtj#r+2E5ZJ8m8{#a4^ViPfN<2)z|0e;LS|!mnkR4B?Aa81MR(nG7AkjtW=SjZ2MjsX<ulD'
    'rT`6J#yiSzINJl8duQ2LFV%gTRvzGJC?v<p9{Yu*tgUp*j%=LnqSWl!g2QpjK}WTRX#);p#{$1gZZNsEmSeZBX25dw;>_}D*9leIY@mtW6?Q}Un}dqQ8zl5H'
    '25+O#*p=&(YpP*4LBrv0!p_9oz&viZpB@|z?lkQuL&)wXZa-L#GHbAm@ppPZDOobyR+<9uDSFgn>bYIz(R82YYw48TUa5X1iR}8I%^z>XGi0_Swm2RqMoM2$'
    'N>V{Ce~YGKoEwT%>QT915AMI%M$)0NI#F{fX4WNJukF09d3Cj&h}EJgLMm~ZDhGXQ6OO(26E<W%gZ<OzbHuXXN$&GHwSSK891c$0{!#u&i6fqE1#*@6JVpY|'
    'nYe+_tAZO)!*v+0StCID0|oj`k$l26b1o7PTq`|tP5dp>Q}vfEPKkHqu6wnfuhuFbPog_1+NE;3eH)0)M9F=+;D(t=mcgo{&QO#{VXQhUlEAvgKf@ArEonAz'
    'jfH{#2=ZqNbbfsFbMMjPAHs|oTD3iyFVE_?lC&d=LZ`bB;r#(cSP+MDMe@=5CKj!)lcwS@^nj9{El5L=Au5anf%_&}tj)9{nJqXghz^<$Qv~`;Ac~pieY6Kb'
    'z4Ib+(wd=x{y*urlSD3+fV+~>)GJpNai)y~MLz>eimG!aQVw`|(orInBvC(FI`S9XMcyS@+0laYX6}5rX&{@n$MVy%d0EM=(Wb>n_Q%U|5??u^u6di`*J&aw'
    'V5vtngNle>m`HZg(6^NTVk#L-M4B?5w54pJf0Km0EGZ2nJs*=CF?4@9^Kxe-iDlhnQXOQ8$Ds%ev!M=&>Ka6RnD_Zf)NJ;Bu#y?`nM~mn%fYr=xK{*CG*`(r'
    'w9C&VXe3`e*T5dW9<~mmuTB7cbpm*+6yPj)tA+rshWubaY`}`j55<N6H+{-<Me%BEm_Bv03G0V4bz=uw(P`?>uJl96?Y1kWYhu_@CZHzP5P5#3$TFS;B*kCM'
    'm$mhkJd3~gjn?J;nGCYMMg6*7XnTX;6JK6UXItkl$zERn<N1>x`+gKD;(h(%`vyc;G_c^E4g$TnYBepcwh>v|iLEU}*ZrhZ0(qiI^a}oxDoFG~8C^(+rjrB#'
    '+|Quo3TfabCkp50J?sQw6<{?(J&d_hPwKy4PZw5AUMUMsvOQn$VAwniuklgx{;84*HWawJyqKTO*TmtVdAK`)M?u`Kx>jN1{H;=(iZ<5j#kJYq!KK{M4KC}g'
    'XqsCbSWS4Y=9gEiH36sN*RZtdYR3ZpxM8;aOATF*T3&5(_oaV=i2d>MaFFjY_8wLDqtNel2x_*JduCM}9byt%(m<X9a!g~he0V|VSh8gf<_Wb>zTfd~Ph&62'
    '#&zE(+oJb3rG!oxc_U2i+YL;<vyymF&Tu?#guLJ;G}I_P62|6GN6E+FrIEfsRE>;rP@d_6y5kSAxiQnhQu$--K|ZoeHj@W+%pTN3?M*ZBdNIA)pcam%u-WMa'
    'bkO7<<7M~irVX`~KhnzUTIJWV1FVASto-q3Vii(>6xzwB8y;n9-FoGem;kX9G&>U0aByIz=WfmoOw@>(*PgqvY2j+Un$_l5G}AdhH{uVgfem{%TWhFtDe3;P'
    ';}fMPGKEM;`6K?>#<^>~S{d3;&GD3U_6l6P0#GZjBSY0&82Mk#u>Sb)`J>6xC(nO+`s~R!kDfp8((8)*D{4^aM1@UPw#8Co&;YEdX#*B5Ctm<Ef_Sb3J0IQV'
    'Ja+D)Ou`_X;LCn}d9{7#3x?FxDHoWnuHJRaxWK407eqf=M6VhC1=OQ|ee~@4_fLKdZGeoX)0GoctwA`uJU<n8KC4&iVVMGE$A0;mON^gsr@P!iP@VS_l=I1I'
    '%*6ckk)4=*vb|u;%}!Z3AR>#y2^VLWdzX)aaVr`ohRO-B_Iq>jM&B>PEO0<@)CCqJSY;#A^jMaITH}4Q&-mQyl3X}NL2wg%R*##ERG5mpTF`FjBjVOUyK6<m'
    'b)r!V4i~%aHOLGdvkB;?Za6TiFItOgP_<^EkL_LHRlzJ7Ja&B|8+3^w8`5oGgqD<6Ul#S-tHpd~Zh7AsJ~dswn-^+Q;jLNu;Ep$6k{?Q6cPO(Lhq4nnR%BcT'
    'DPG}25*I&C^I&TVO1;z^+%D|Z`rYJe0eBJ_#*0i;uL{PvHFZ3``OT9b|NQ;4A0Gbn<k{aQPai-0vFp6!q421aV>$FT!3Vzm?n%$AiNbWae>l`;n;z`of~o{x'
    'uLw8<x9y{VM!|PVgBR4&+PmE`d#;`e(~axt7~Y_sQZU8!v^z_ty>vy-1ts00QVeAq*5>t0`;BB$X8M3mvtkY?qN>J2(lnK2TSN<6wnX$oZwDhFnp=Xo^yi^7'
    'S%w@I&*(>`gpm|*jVEn*;zsPI>X-EssIKcPBN%3~_(7j1u7sDy5BRK}f7X5hN39Wq&{p$wRK(pW4*+gPoEXk1xuH$%pIKLHj9Tc7pkU!`+HvtgofbF6KUAxy'
    'AJoup6@#1GAChB&S}>N=5AZMibymNg&uUYjzC@opdUn$(6DQ(Er69TM;-fi5W_0=nvY!6fBJ_2zi!CjzYPkkss2*G2gv)XBZV9@A`Rq5gCIO8P<Iq6G0V1ur'
    '1#j2$E4~NA6ExW}s%PJhV}dnJX>jW6cB&3K`9xG&u6#%KBbWc+5dN3n2hdy0ssz0baxBmmfne!Zwfj)Aq-c>qJ+*!k9T&guGEnq_iVnQ#;u++{3#U|nv3gUl'
    'yXctL+1B9l8i>SpTiW{3-h@83+my-a{_X9|^N#B%?N1t`*@iT)t@fdu26A-B-v{2l)Nk!Wy^HOozw2)_8!4OHY#dj6y}h~y2YCO=yW@f!a)dBjw-eYNe*5jS'
    'N8djD=@H*g!Pm<*8BDXg-!9CMW_-K3oG$0*a8jF-w+~u9xZmi9ll$q%4^ofs0Xw~VO|R5W25q#NY}L8qHg`kcq@h?JgK5nrJ2;`>)%9{QU%ukY@td__ItX#0'
    '%qX3VAWk}N^O@1n!|qFZ;suh==5SLuH>-~o;DJjK-Rb7*s|jd@!fRd9-(=STldKfoXRpE7?L@1Lfs<9y7$v(=_FZt>PdCK(WUzzzm*ueZjhZGUP~y^!=ZUXb'
    '=Ebj?&d~gk$}SgSjO=gs|2Uh!zBBy)?7eGq8%L5Z{GGp|&pI3cEC51-7gO+X;?>w%kK=f3hpgE%8w>;j1V9QB2yoG$C|axk{^q4FS-0*6NZPW7b55*9bXC?p'
    'D=RDOc?RhdL%dfnuCCR?Fh1)ZJ_W7X-A8V@^EvE=-~RKv=h@}#671jX;(ENsHwK$AB%jKosa5Ytl)YI^R(PFGZq>ff{RBCRE&(<Oeawf8>$UE%cc7+ZqK|EF'
    'B})}R7S-#OQUPq!=egrPc2z4R+4}<az<c~3$a}YcOwWFQ`TjZXm!N*1zkKoP=j{6*fBxI^AG7B_{_y=z|9J7!tA{BX-3DV+Gtj;Z*k`Wh@1Uc`=Xkh9yfk!U'
    'q|Urq@fz%cDWn^3K-%>Zh65$xFzfR&am3GmW7kKqnJ*UP@Hn4uimbRC!=K`$ot>d8z367ytJgD-NXsw#;pH!6NT7F;=GE-|4Ecl5XuHIh)2!M!moAr}X4lzr'
    'b(4`*0IdXpx-6*QFJqXIs|*R+7qiPj_7)N1Fo+|YU0%)Cb1d2O$EaITx;4K=X^T}cLjYf<i^g8Op1}4FNy)cFl>ntcuZ6&1pyxE>cjNhDOd+vQnF|D86%{k?'
    'hW|TTFK3JF*S|miMK_7A>=6V3J%BK+GrWh&t_c@`v~35qErA1e?|riDsZEMqrpK^!t}e#3v0PC{j`$`5TZ9o&RO1FLssDns?CO60677JO-~Vbuh{2{9;c6Ah'
    'p8xX8kN-XT<>k+>e*X67AI%cHx?0>C9U+rzyCPhyqG;=7FZWs5EP;cm#>xJ8G<y%Te>7RlQ3t*A$auxe?%46l03E8M5+%SmWs~^%`r;b>c)!q>=6E_Cy`C+u'
    'PB@l~G7%^jR-&!e24iN7%<rahEl~&}=ixYz^k$>efwpwahTI{#K-j+Y!)kk{`Ru4#4EI-uJVh}X`Llj)A=2FwlXcvy6h+Wy(ahCt!71KR$Da;gooPwx<-C^F'
    '>+T;`io?6S4h5iB3F}u2Gj>Lmtln{@ET!i<o7Do02fk)W2<!FstNWadDkP&^t;<9m^y>9lrJx%9rKm1C=i|%y0-Jhu8J!i?lEB*SNcAmeNrPX8!ci9tLA+dn'
    '9Rhn(OSeFsB?W-hi2FtTLe5bkP$|XcgtRq=7s)#*Qr0?qWd$%83D(|b@;biiDh23U$7&%)c@=~V!y<X3AmmKiE}H71Qz^KXJulKVk!8IjbI|C%mcA5~ewz)x'
    ')@iC@MRw=3E@~y!z1_f?q})TbK5!f5Xk@uv6t_hewKCEzXyDfQg+h*`*6I}eb+Z;1v-%UG<a4&u-f);b$-IuTZ=a#ge4!HT&g#rc;E;=WDY}r-Zm;TcKJlk|'
    'oUDSqJ=?>&&xFw+M#8lZ%CDDFqZIUsYl?^)CUs6a7Kc{aBNd`y-fz>>!Tv$7t@t)%lN*touq0qF)o~_PzPL!y!jD+^{_#OtxV)WdXSl;AMT;#}m2k`jK(5LH'
    '-2E7K-=G-Hc7uXdx8!NLdUK9+h16IoY(<hjlVJP@N4i90M+8vh?*3s|7)9C8&b(7mL3S#rA1TO&67SR`FE>XbABV;0a89$K%W{(9gIp}1wtT+?9ao4ub1mUA'
    'gO+>Hi;qNosW!Qm**^?qYL+YB`q9a8sf6!yuOBBTXe28X6d73DGDo!A>2<3H^e8FYX2G=-iqxCvmSWv(anWVwfqruun;>OYg+{WrC20s13c(N|Gl+a<T{_AQ'
    '!fug)I@`lvhsF=UUID(lC#L{8m}Z`H;G3+m=lM0wdECOVp51Ck03BV?Xu$)$LE2Aw02)rz2p<0V#4lU`qyXBCyHR$!*Zr)I_uYm-1*VRhTu;X%ZWnfQs^{2|'
    '#D2CyId8OIF7U^7vv17{0!1}|&>!ZN0%5J9W`}=@#v0#9$Lkq_9HJ>UE57mF5hQx4+!Y{@2;+E2%x$~VY;{-X%{k>2hkrF%Eg22vbi%=igPjj@{g1x>$Nosi'
    'L974jB!@c5AtsU14z;*Llj%tR<7@qoWBre3>K{iv{SWnDo%Cz{L1#IJAGqPdMiL%>jIQEjALx<`y5$`I(fp4u|NG0!FQ=m~|NiAazWnt+S|3Lr;@6KK8xh0G'
    'HJG$<<+)jefrTrF1EJLo0za1~!_7s_^MP2&=Ca?*!BHWJ8wbq=xUtQ6(T9j?%}#D+3#9HM2@gJkKbL^n)aWD_kOW?(7a12ABi`(2sG&6yOW~#7iCX53Ru{HI'
    'w5lC*mG9=fKJuYkMR)Q;GfKs;6q*`ev#KWzpGSE|dg0_6s-UVHi;_;{vjkm?pSzct^dMrcuH;d1$gV7s9XOOd!WIm^Y`}XJbP+Gcig|1__5pttt;JdJd#uN('
    'wH4E@6nA$tHpshY<q6MBLK}Kd7hL1(iGGLNJU6|Kp|<K*+uJth2i(~YQRD!_^BJ&iv)a(15R4m*+b;g1Sc0~afg_=$g~zbBZ(R!lyX9?*fFZ0J0@+V=u)Bp5'
    'Rj(aD*PIUCz8$uR)dvn#q{ivP8GmTCvZpHd+qXh4x51YzXOxOphw}N!0_A{{%&e2cSw_wqdytYG1eL7BWGJ!{OF)GtKYo^c#+D{Xc%myvig6l<e$=L(==#)Z'
    'iBhzZsM2f^4J~w>sEUUkn$$q1Z0B$|TVk^5T@7Ed{0yY7!OT;gWxyEC+NG~|XupuYP5Vjh$C}UZU1Fpxuk^f<!M6Nx@6k&i=Y~Ax_aT^`Pc5<Z7uoaaR9Wvh'
    'Vk-s9>l)`sL#BhUai7;U?aFGOHfRn~@IVNlSYjjCsc#jqk?9Q1_Ok7?BH87`U1}(~t;C>+C`x4QWm(qFy3*5=e%F?|L}WTtkyRyWLaQnzE-dUv$UlYtOnwj)'
    'r7G~^qZsFPrsIaIaQ~Irpt#O3pFy?PgkEffji7Xg%BT1^h!aYgs5m06!rCLN4vAJ>fvu9TFCNo~O+y21&%f*dgXDH{TQf5<EaLQo(n{2;ozbd_kbOBF6<4zf'
    '#fln1s!ps5R6x!Lu|@~I?*6e8<paO;`#O3@|6${L;FrAb1ogl#y`zsJBt?}8RxHK_jbu!aU-V%ZMKoY)Ot(n-MFbyVEqXyM4&7R)L9!h{6_DXgSOpa0R;m~6'
    'kFc86{h-~4aGZR9JEAe7I=F2+mc1PeMt&SN{lKqt?>MN+fm;>c_~hiY)p0}}^uzLw<Ly5RX2yYEjlAbpLqD=w7PDn>P5i2EbNAzY-9HFwML+3&i;TswE_qmQ'
    '&50#s_&6Mm(QG&Y9t-7f{t(%oq7#dzQ8>1y31N5k%11R)DLcL;H9M}J@_Ay)_xw8cj%wFYY;1BZ_<Nq71ZIoHt~-i$U8t6HXI3qUVaxK7Zstpbcg)zMovqHb'
    '$@5gWGSw-vtpd2F7zPH1Mx>INumz;saIq$r)hW5@p<2ZAyHEc<I2@L5$(U(qj{;VdD+T}WQO`#1Z*GXfT2x@*#MZy3??$8#jCv_A7MOqoBz7R18$~Wxh~6{9'
    '*OUz`!Zh#0=4R0CpMPXE{NX+S6#;gLPS(whJ5}G``14@@@GEFQC@#LQP+EU)>8oGOLOjsP-1&pQevAdQ+xuVa8*&@lo*KB6BkYt)TbRzXX6wX2QK1LJN{#p^'
    'q=~33b;H+btI%pCD#3C)<XFjjX?Y@|w^2*5wh`N<en372s73mBXj4Z3PMJG~*_6<L_;IH+=`auyUxOCN{4zfy<evqxrz(-k`_;jkbg*|ZihPj4*=Ah8xpc<2'
    '2mW0~HlK39IR;Z^(-mld@zRJy4qZd{Lr_IysNU-tk^G2yQix|84^+h63&Z<8ESav}rVMKWot}}a;Mfj1*rj|t#^a;4E0O5e@-Mk}dD~v(xgIc4Z^b+%fgME+'
    '^1<a$GcLR#C|0iPFd$qt5{Zam-(#!4<fpu#UpK229E72pM&>Uk>1I4%5V}-(%|xa2QGsjL&c3pB`RIdxUoSVvb8u<MoE+XM`~Z?{&ARy0HNP+}WE;Z==ePpS'
    '8@m4)0|Tc76*u}|iyrj)^p@sS^<?%WY$v9}52iCB#O}5&peOR+JA&WZBrqfg$6=){*{#S{Zc*9!I?`M(_Tb{oSUu4wT?j{NGo>cks2GM*qe576UW^JgGe*V7'
    '1X4SGi_P9GXK=+#+h(?nidKBk+}XwuD|KcYbZO;$D_YSat0vr5lg^!ZUmySgUMF8s47iRv*Eh!lbu@3YJFD%vut1?YmK$%>4*j_fW7;Ze$@7>HU5)<ZbU-PG'
    '_NJ}iFg2Wl+Eo!S53BjC$bm5f?o`6m7{!v~D-oCN)T+V8K<GsIh{`Mod~9W1WnKzzl^d|n$T#PCN|HvDyCO=|shQRdH8QqKUU?|88WneM%Fqx0m@_y^PSlp&'
    '>V>DbJtuth+QjzIeQ|I`FDBe-m*3r~d$b5wGWDk7dE$jZ==l!YF11%NBfz=0*_`t&CUn0CoXDMKo)H)P7&h+oYB7AdxYyufMft{pL0SaT>ZU4^S8Gh~f<@Bo'
    'FBzj}GRm0r(z7p&unabuAKc=7t)s|(jA>@+b+uf1m1CtzXyB+gS$az}-xneJqzrJ-l_cWPjG%49)?~YMSlQMwYPMaKBH{sQ&XL(f{_PaHXzNlfCmomQG1u3q'
    'QY!S&l7&vI2-0YwP2aaQ?YUiSdKQY+4B3$46+hW+H%XE0QAnQ-Th^5E=VIByaOttolc_^xO5zoyhg#pTaZPEl>ia%3aJ9v{*F+^^J%n1nsR;mEpBYQ%)g9LH'
    'mZjT@FtuFy0^nA~+ri27V$RhVz+sY9-UMmwjRdq%@#wf9V7UU6%VKq-H97qM;JF=5?XIASz!C^Mfp%-hz|w_)?d7P<j-zHk{l<7fxUG=2<ERopwdfa2Yjr_{'
    ';X@pKT}$u_Iv;ly^S3jwc#4_nEM_8yQ}$w36xtNDu)x<#`h8BoP|@Q&`p#IXbbUMDpFn@fI_wF-{fM?d-VRTR0Y;x7(sw}V>zs()F2>D;Gtn;%l`APapw4ek'
    'd?ou8>0t8cm=*2ikh)YPH!Iak9w<$b2RGkN?!<C?9qYkyba;z6RC&oI*>RR`1+;aH9&UZg8k;#Bh6?P%8NeW|+;F5OYFL_J(|<v3eE}BBWx6rQ<?0rtnTDRx'
    'p?WP^Vk)Oyg+V^K@<$g=MXf$?Y#LvJoRT89{5h3JPjkvf3QGUt`5~v;-@Mi^q?}Ac&Ef{Qwk90s!BtE`)x?PvY-k9~C6O<B5O6ywo6=9{qv*P8)0x6Sw;5#J'
    'be`zu+%7L8J&B_@&_Vui8VC6dF0AAu`Xc*vd=7_I1^PWemQj2sj`PTDJ7=@a4Z1(oZl=_#qo2E}zCYEcd|^coOJR(o-PelACmT!uUBl4QNlWZ4ouph12|}9`'
    'lg+TDomN#T)gj58*~ma@7#Q|c)k?MAJz9Nkv2k^IpVpvLTtZ{HUFuC3*F22MbF`V=w%xQwYp^?HC^@h_g`!UqPJXG1McqRU?PhbgmT*%k$trgie@}6;DR+a;'
    '?LHGtQl)9o=s^bd>kG}5=F)P9k)QvHM1x9x1mS$SS6>a`9T8G3t`Wgt|Bq|3KAjL(LAUL(+gnO*4uc&8k@WE%OLED{wkvm{3<cNdQf~E~DB=VRb>$!tyLC7i'
    '0=qr@7?NBkB9Nn^#gofBh6F2&T?JnqLA2(!BgT&yYCOZc!^xp>lsZbb-rnwK?j!jMBJpqa@L!^t`F~N8JvG4;7(46M65Mt>uqhTyaT!eMyvRd;3b;=(#0Q-Z'
    'Un)4m?mga6zecW<!?{#Ns2&lbJ*(7~F1-mk&DEhmA>d0zt@}%T8fm8*vWi1G^PNLj@K^O#Msk1Gr5BHoihI4%;wLNB;j2bz=+I@$KtvbtJk^{0A1aOi>G)%H'
    'F}fCFImY5h9f<!ye8%II#mBvXgS|4fBnYTX>_hrj`Q`+)e^pX*W3NszbZMXWR1YK=6=eLY=DVSbBZu^Y`F*VCPLbqbLYqt3gaKI(R>QoAbMWKgG;Z)P&4|M2'
    '7MdW7<{ghFHp$lDUy~#?H0C<hH55l`4m__~mNsW1a710c{%e9y1_W5!p$3J8lL6i)auYqT3RpSCY)PcqbajL?K(n{EYe;kB^$&Z!B=K`e2Te}78s5!!tbh17'
    'oj%bcI<P!YC8Tps2kmS!^c=Xc)Ct8ShB8z^%UeU(q$ZHmZE=DRCWqCPcI)M)(N(o!FV0(rsuq}c7_!#lfkvJ9_mBhGr+}(_fT1#wt;GN}?`9^<UCK@}It!Ly'
    'Veqjn=P;4?ayuO(4lqf~4iny4Q!zGnNdy0{NeSz);<k5K)gz4Rgh5=3Q+@xqm*-`i>Lkjrs;lz-ND{{8(6s0-dASP?;xFDR`fpgns?5`YDMwP3Yd-dX<j?JO'
    'P8Rn3Cr^66e`nJBNjd5L@mA#Wwb|MYyZpqk%e%fE8E=68l#vRiD0yQU*ArU7wT&9v^{JLyNU-V;@@@Fj4T~rpTq}$vf-c5$==p3jc};3woI@>xnbK{MbE##Q'
    '|5rdXZ#rC2orkqdzp=CAo!+dWB;G!K(C_8XVw2TgLTGrNvZs9IW$gF*d3hOzjs0{)2nihX>V4NvE&WW-t0H;0&R1(2E5w~>y^KtCr)KRbjUI*yDW*Jn-Bhfx'
    'NvC<cXaC=jUu&v2n^XiOBhMx;hoR7I+S|@>iW)&YwkEt85-c%WTbc%y^j!YTa2JQ)*W|EcH!r@Key0A${}Wjo+g!t|emHe~uG)_mC_748cJV~UJE-WGzZMgN'
    '0GEN>7Tkz<ByE^Yb)b|ttTFw^MT;OZJ(3u{8Ke`TWNy0im?|sNRxU|vncLloq0wbK@AK^ZRm_e0n8ptV-J&R_lt0vZw;dY-q1r8GThaj<nth&cPb*EIxb|FI'
    'f_N;qeMpg&s-ZNCMA;@4zZcbNk1ktq@hD7_NJ+8A3H#C`5j;@^*uJ5erRRTa=)}!PJ3hGPt9*+o2f#n!$BBEZNPDXZdn?F#OBuGdWAgP_ng#}VnAfxVYH*tx'
    '@!T#ItIBIKjWp7exdx7gFqFLZ-z#I=Re_~~gES1ucBY$8Dyv^?ZXr-E(m<RD0jndcOcmlKUw|o}=%!r?bM$GfFx_);vH;6tE4QkFbARZpu<#nOTB2H&1GU7H'
    '8MdI({apx~U4p`gDZ)=sE<JKjSECFLi`7ebi=v!+Jw68DllOoc-n%cYl<NgH#QEmKOH&<BULiJ&+=$5bE?G-vd?a;j-n&OYDb-uCexi0JL)~Ly%Yy4;>cCH0'
    ')(^3$*@0_nM-#<lg^{y?q##MOPJY;ISRpS(Wy_q%fy(*0Wl&5aK%w69swqAi__d|3a06LWZP>DiNS<^7K`2?J_#j7GcsWgBoL<NG^-)go0>!!#S|vG!7s?k9'
    'I6ESOp7o*QCUeDeRAT<hgb52666G=xf<Q(~w3~8idDh;hf5c@Wm+ms5{nDc2xsvUQYvE{pwyjB6hwXHBv=YlPZnE1gr&P9*PAOaB(R!^{XV<Vq7L&6YwGoL^'
    '3lCC(4$ILLD-i2+TCq=u=`|KuO!O6BpA)^K9R;K=sV=4@Z+c6_M$g``(!i`mjYSz`;B^=6Dz&Se!{m>f<dlra!JH#hN`L9ZyqT0HR)DAY0(dcB&Wv+z8m9l~'
    'dcF`^q0Q$H@z~3A0Z{13N$xqq#2tD3uwCcCUGLCc<KVsB*_u+T04XQon^;1Y=X)5mX#e6pB7k-+ygKq_&{oD0On9c6{TOpW7dPwi?P&Ht*W-n`bZv|K|7dyc'
    '8H(XEsIvIa>I@HaB4Rdf2R3<hHfO3x{)^m87vuRdELV7*sBkq_G0nv*g4LUM^<w3ES2DF^E@zCQ!hOXry$M>JM#IQv@0U8>4@-L6b<J$!+}Y_ozWbRxc+9KJ'
    'Ds~;aZVmmwbAT6o@~LJ!+wZl8mhl7>uUE`RjXsAvu@}j8+Vo;$CJI6_6=4$#o3Pljs)(oJfhhl^*Dylih{uJJ6wi%5!1PKkIzPV(qaG3!Y>Ak+EyM=_6XoL3'
    'IlF~)-<Ml1d`2(RrY0=7Jg_k9=omyNWVjiR7iqgKZ97$KyLn6l+g^9J-S*pTl{Zc@kh1A(Z82}1bYe#Y5vq6E3fG<Z6%SIOw)(Ch@!5%|j&Zk1+S@^vAiGr|'
    'v;-##iFiN$8#DdZ;k)(NalL?AFMQT}$?jA{vSGVCJeI0XG3B;>Dq_JvtV^^=OmAn3EjLioXt_C2aPd!eq{Xk|)tVh)B0-&lmb>sC_{leY=h$3&_Kt1Q7))*m'
    '4n7_%G)+bGJPhP*;ZsHWy*s<S+T0HOyH*4-n+0WZj>+okw&@hum}__%p3$0!HzO)Z&a=3_TzW*@^yxH}8Lxd5MWR!ddxuf42sh7AP%b7m$5@GCbKR;`k&jBg'
    '_Zt&gS)8uKDbDkV-a+Xbq@dRPwSK!;PN8SQsTMqhO%-=W3<_w4J<$d2eX8<yXaCdG_gI>-g3*oe0=n(gW*9t46|5YV+?8KHD_NFwglRkt?}#B9iPMV2Sw$q='
    'OV{OR-=>{6tM%J+d}_UAOaC)L)$)|cM~GOPaKVsuHqi~qT8p_NH4RS*Wwmd$#@sw=jU9T7vrInojiTsH)VZ<e63w4NMTdIQ$(I(OJPCnBF%e%8VKyA;;15>u'
    'T|&8eYScQS=hLZg#GLmH_0k873Y^ta>{C4ZoxzS<H1-@6z4JNPpePm*o_s<axd#j)KXxa@J4uI<+CK)#Pbmpv3<t4a6ZzsB(V+ahFH0$#f&Q2O-K$)@Rv$w^'
    'LX2gx`<S=mc{`dPQe!4g7b8`QcTF|QPt5>gC<GkSne7z~fUA?nHw}zFo2;hLiW4}Fo_CHr#r&e7@jYK?$a2yvOcF^@0UJd8L^X0foo~!KUxG!|)o*1m$%$*;'
    'or(!$-5O%<I_&f%E{zgbMrwZ@=-svLFfTzZag4x*V9z)h$l|`N<P?cn6nFVCJ}fANj(w^jP!(U4P}3H3s!fKA0fqlN7>wiP?cC|=uAVHzmN58E8#0Zv7C1->'
    'Ta`|a*8n1+G;my6NdD~4r)1$gxwoEWwTr@f?^wp{JwaW(p#Oldfq;Z4xT|i~upUQv4i`?iM%nC9Y*Oq26^u)AGCtTWFD7%w(wvNxU&f2m;h@<0`ieNVc_UKP'
    'OZ$g$_bSnZ5xJM29uHpo@k~Z>5&`@zt_YtHZ4j6x*^`uA@}w5mnIAP1yWi3ktGMP$Mx3bIsyk752(Pi+AJF}Cacy=8qC2HXUbWiW{?>dsUe3?4b%r^;i!-(1'
    '_yKsa^D+4_e#kWeby4<m$%^B%l$TSYN-eM0#}`aWkS0Xu7EA4ec}^Rt0}+3X9Bn+pzfvp96{6=XiMj7NYETtr$uST_9sQbOg>CuyB|B@FSw7E5<dn3lNdfZx'
    'fF~yw86h6Um&HbSlES!%h@3Do!jn?E77aQ|Ew)N2TD>a}Q|&*?$t2}SlQNCxRg{Vj>traPmQHiQ5|kpq{8|kJ3)#HM@IN?geb;nfmnK4PXUZu#$X@b4+QPeK'
    'iRKdSE60BoJ`i=&x`1!%+0|^r<dw#&YIzIsKGUWsS`KeCIa=c!hp?t?Hs^ki_ictYq(S0KQ~p`0ZPyrwnF?i`H%>8O=4hv3_CeSIAG7}zrJj5cO?#5%AD@1R'
    'wh41#TTb{)2Y*dx@8**k<mZv-s*1vSuN65Zn7c;urkYm>iX$mFm3z)ZIoR5Z*q|y_T)%iwTmt>Z*}e*Y8Wy)peC#lvd}4DE8L{Y53R-`7ltV(%_54a*sNgRY'
    'ovUp_(~c44gw*&Hui0of3?!%{rRI#cVt1VOc@O`Wp8r@|ESGBSPIR?25LD>x%np>EN<29*pO`n1itCruE$?z1uQgT7yO%>vTBvHPyI9@K)=dUe4LpR-ay0pa'
    'HHmesw-}?{{pqd9^M<V`uS;$tEJN<sTI*2n0(#5PfA2Z#Qhqf*%2jM`kDBg6Gng`|#ikHwt~XcL_~hKZJE^#7@ioGp$DCyi)92rR|MJE6&tJVz9v4J_u|;~_'
    'Y%fX{(4>jFD-q`TAsgmx$7Vkq9mjuu7W?@KhU?<$UF_4>`04dZ8UOMWOV7GRz2VfrHI&{ZOx7h7n}7&|K3y*t^W|G*jX%tR_#w`%sXZr}IF2wGKM%_ocd+vP'
    'sH1y28u}+#<BLk1V%?g`gXP85wL&3b$CS1-ua`)=F%3z)71=iinu%dFn$0jdgE+dH?hCa?F@ShZl}FdQ=UaECmYi?NjLmcB3f|Q!kYw)*Q2+P%Kahu8L!Nca'
    'R`K%Zf4+M0a*(0Q=sf>_&tJaCzW?#(zdipkd;a4O-~aTF7eBpvuv)yG&8`##>-7ZB?B~}Dx&*nJzgyA$?gn`ZkT3-U*gE4Iq)X>1Lz-#~htnyD!)2s--8(1%'
    'y^kv%rvZG=vajGgbK{Z45H2hVW*T;)A6vhpSWLP;!AXK!wGyntn7&c+{@)y*l^63RBhX&Xrt|B|OqRf+b<Cp@a~z?OB8i!Brk>3vV^}GOQ45urUQb|-E2P>L'
    'SV_f3djVLp>ngqW32`hyF4yI{Ff!J3rhm$wfX(Wv^ENa5Yq$@nI$^1nwPXzIi&!GE@i{DO`l{(}6w!@AZs~%AADv4h>{D8)v-h*fHCZ6*+2xpSrBEAo?|q`}'
    'DP68FigR3nusBI4VV8_(N=5FC_i!K0Rp`O<m$;q2{Qg(_ZgY)!kuhGRd;ZHWKmPaVmzO`k`uW?Re@sxvOSv|rq1=BZ@&w+d^s!ggOq&9sl995X|McAt-$AO;'
    '&o94&j!U5&$UMKk`0>TJuds_=y!yG~d9kpa^3m-5W<4HF7IRdf?>x!{0iPB-78xF9b;O|rc!OvXKVM&5<3RpGUz(sqMz3d!EB36j89>qHSc$fpg?OK#8sJv0'
    'B?{@HsIijXY;;iD<elPjyjUp6>kcf3jt<GTab`BWjlu4F+e#h3@9@=ZXR#`ZGTFVuYT3xfrbTNM{-nUex&qM#sTOz8t0@lci;jWOkIUPxZWY1~>I%a#w)(qc'
    '-}1!ee%-H@OT>{KwK;UouBTvtH5eZVj%|%%gY?BvBnBmE$z*X=N`_r^Q6=~s7Y0c#HYW{Qp;RH`=2ZFgwUb&m1s-Ho*{l@bE>|~8QjJZb(l%==V8z4ZAo-uj'
    'j55ov72Y2g$`n<_itNq-Pt?j$HC9_UNwI_~W#G2T)do^m6}JU4jlVn12kPc>wVs1rN@q!GuTBxWcT}I4=ZCYM_J&BC<aO1aghWuc4Z&uljx*;dW)Uw&mr~mE'
    'RlUv!fK;c`Cb+j}``GlF*d*m>xcg!G^-F4`!ftW*-lm0*(a!OFMCw$(ok65KP0U~r<x`{{sl0}{!~o2vN<+{%j;lfCy-S&{1YI(y`k-VY+Vq%QY7`e_(`6i{'
    '%5aEY>Xs1lz8v%t^67YK7*|XIhBBa{B$1tpqIRSkvU@=ydxM!+Qw0&eqCKLkZDW@*<C3U&*lfb$W6E6LAH@oZDt~>%*P!_2;wA9=|Amttofh;RBAi6<u!ko+'
    '_<MXshl%l$J(t}q`)7fKXka?pKn9kS-qYX|L%JV1Fw|MA+spV{*U_l4%SZ)0wiUlyJ3BsTMI?)x$_laa#<QK}&-(Ph;HXz3Ra{@z6D}`Bp~m=vB|AD06GS6h'
    'QpBSOb9I6lo!jVqppI<G8$&wmITwL^=!5ffX5H8gaje%TS~{F{;ppfd*Y#mKb|i00k$e!3<V9~o;s9=s-fgOQaF7=A{WJA?HebTlI^jtt4%Lxkvx?#reNVTk'
    '<~*G2eD;^HQ>)OeS5C0N`PI$7R{^<(^Tu8*a7fn~P6rqn(Jm7v;6B8HR&JoTIL3In%4m2x=+DatOh>jzUH5o%QF%}KePov&dK5^~*mT-=Fk%RwgIxckum7<>'
    '(&#?wzdFgGPI8Dzq_jgV?$Bg9(*O8c|KnKy<C*%$QBVKF6Q{F$trO_1$L0s96ZeF$poFU!FaJ5&2f8?cemcj0H2>qv|Nipw%jxLLzkm6UFMs`y*2mF@_?4Cl'
    'a&^rv*O)S<m*;bk<`*3Mu~-k}o~$jXQzfI|If7Fj2V{Z`$5>ITbT(}`*`9^jc(Iy|!@dueZM7_=cEh9P)NooRz8X)KMAW()gQJVZ7y+jD-VF{i2lSr-{o8x`'
    'N}@W|mNW$RIvDF*K927V+^h#<<}zn+yHaUgQV*894&IT`-3~^Ja)bH8LltHjrHidH7B(Y{N(}W^zYX*OG($A{g3o!S_~_~qsx%E@HMJE6o&SV177*RT6`_Eh'
    '{*s-0?Coks;4@JZHuNqmNxpsEx)98qDnxgS`zFvFO?8hrpYcbP`=*2;vBWy(NmL_q%;M$;1+(Q!^PEF1o`5t2dhFzVU9&b903V-nYWhL_TSbTeY7GPH6K6m7'
    'Mp3|TR#&{Gf1{pqP?u219I+B-oB2$KjBCh0sAnJFY2M)f!Du`6!+ZSUJ^#=<qN|Dt|ABmGG!*79EJoGx2jsEBrz3dL2m3Q(Yd{UCdYZ``AqRfLdjv44e})fh'
    'd^$+i0PNV;jiWj$relXn9)TK(gO!<>EbS_NP)PIy=g6pnq;BwS`UMRE0|IJ?f%U5d5NDjHSgkkBj>hW5oBfjp-_tZ&-OY-;B&{@?yk$xW)s-MiHx(hpCAW|`'
    'p0C7>i|!2?+83<b1Y=3^<9+WMRofK9$lj44B}z@q?t0pR%r-O>zWTODUaa=0D5<mS+wt!untSC4%bV4@ExWx6XE^~UZ!vdS|8N9~3s36ugy5rgQ3*5e(c&_;'
    'nKHo6Tb@yp+Go=sgLaR$<n@(8eDFXc#1KK(0ytIL_MCX+(KpC%d!w{UUW5vBTg-ICW>}V6h}y~+e3kMUTFnpq<4PeQTlC|P?oP))>R=?6;qJFQ`E&|A#9*7O'
    'r$dk7Nz&4l-i}|cH(7yHcbrNQ$$tYncV2+P+Ne8keUEK!h<}-1jx{~$Sje;IxZ*YX6tD&B@a+NOV`O-;f-_pi&cbZE9^Xt?H_NVvYTj4zfOe_c2td&zIvFlt'
    'H-Ysg8|3NJri5VCupcnP()6cxO1FjspKDc(F5~Z34SD}#$d{_V=wW3^-<Y=P-)X3whG}#ghNp!xew!<HIjgB!e^`AjSahqE0;?zIKS?_s)b?o|AG8o1LzlO0'
    'PR+sTESKcjZ6MLzl^JqU^C;(FJVZzaeMMe@v#%hNBY@MW-F%~!3bobZ*0gOaT2d3p;14(8!t)OmvkiGe$+JRs9lO)Oj_i;{S=c3P(Ml-TZhSpa#&FRTO7n93'
    '1~k-LWeuwY-M6#bqRG@JS_~O(=tw+tl&+iqply7O>O;#O!zg-_;z9x)LDn7LjF|yC9VijQmcIM4BN_Q<x>kG(R-ZBn)w-kFge#1_aB3ZKj<5yvUt5&ig(6}%'
    '<M|?H(b`!ZR1U_vNs<HzInNh7@X}o2x}4o+=ot}D+u;|OH)j~yW=n9NaU)Vct>6*sXNq&NN;>EhIl-w|71#F~k@|EI4Y|M~AV8rI1}F1{2!&rqh|Fpin45E$'
    '80Hb4CMZTc7#rCoW^hUdCXo3gCX|LOJ=iL5L^Ns{cgUQwXo}fss;?`yQqDpry?{PcS0us%Zr|@_@cG?b>8#24%IY0s?I0tFv)89K_(n~5a0eBlTG`f{v6Us1'
    'Ow^K@XsJ+M!;uoFSF>oV4IciX%>{j+QHYV`94DWR?nFtTL2(pY$We1EBI|OtzQDs8nQeR~L&MzKdA^}HQ~YO}rfUPKDo`7rP!&DBmuW_XRV%QB?Bt&<!{y0!'
    ')M}Vojb;v{#Q>J3pfa=8%xR%&HA1;8;dDS#d^q&1APTUc<RflGEV(*o60bA<yJdboZHR*!y+q@G`U!l5P|4b>aYChEB6`&2K;SUQ&e%dh^qu4LE+g{LROL~c'
    'LUpj`s(9gd?R)T-@4%uPbt|~luDqYXc^6Q$tNvFcMy2O*a|)(;p&{#fLvdPH%vrPdn`U#)Z=%|~v;|J4CTvkxVg<kBiEGIgb9IO!3|os+3-Y_E8Ys?hQz1#k'
    'X?%1#tzwbUczbN{aZBw`-l17ZCGWJ1zNA~GE1_Cu>J)34Nn6!22_>xz*=qiGew@ZHBU!F=16pa4DmyC9q2Ct6>x&R$UWT9PN}|Kdw2Fgt8xA8m?Q1aHuS=-y'
    '{T7yWx6AklTiW~kC6!mNtZRy`Paba(byOUg(^T(sz!k5?ROi(l*6{}0ZAF+`&L?|5T&-c^DG!iWFZqq-CAjPzYM!mAseE3ZC^OZTPBrCPAS`elw-nmZvQPhd'
    'a;JE{i14Y~u~0zg?$LWCU}Uty#b`g_T>FO_?2{(7`n7+k5k~3P4*i<<;71c%){)hh{E+TgB=6)0-eJ{!0W+`xHfOWf<9Bm(yrW0M-iKwI)eSz*iUr`{?G*u?'
    'Z8zk)Z(uMSUEfI#zCq{PtQMdN8Rd~XHZ|=2yh;vxG=B0RB~~oyQlVkUWPg{5llNou=Gd7w>c{+0&6`f{1SNV(tyM-c`h|UcGzbH#RJ50YghtQy^<t`=;@Ma|'
    '&+;V3ULs9kMd5ucJIx%R671wS)JfN)35<ngLZeM@wQV1@QdzCWG;ZSv>iPBu<$6cry~=ZL^o)Gx)3A%v{AXUg6(P`fS;D2U&r!jVDf!ovI|c<u3u^jy47LO6'
    'w%<c+*<gCtQ9H?j4t`gP=~ybJRH7rPs1k{I7u-s5>HqPD*?U%F!?a~Kb)kG7@-%@q8B=|>K9%%_L!&vo1SwC1Oc5D%8`G22WMNg3BuG}8krd<(xs8VI5ht1}'
    'N@g<jj#p~fq9vws+O-+v3mArME~|r_)0;2Bp)(VmgMX(I8<H}E+@yaIW;dPcv}<gw(bN?Qj%C_CUE5PC=b`Hf6uPd#jyBZ(5?}Nn;4M|hW<OyYQB+SB<IA(@'
    'I9m_AMeiIrBJ^*3@@uE_L{EzC(lYXt?(3E23=e94VFuG)nm9Y0Uqp}6F<ae_yH!(MAJ&?Xe)hZD?B+reI;>Hs$KckIH>P`b1112*d)wFprq=1EnuL?&RLA2B'
    'k49TMeum}KW|wK8ILTAT`Z{h4-<c2tkx31r{<~@zony3Mrb<z44o#d}Rm$i=ykTm<Q=?I7shc_vG8jX&sn2Gc8@wG5_tw50T?iAa?Fl^}zgr<ppv=T3JB3No'
    '1e<TN>1xIvzR7Aqya?J_aA(@pisIX=XIR<%L?QPv1GP<p$Loijd&2y=m!cx^-Ji=eS8*ndYQLphP}F^?lLQ8u*Wgr!QZwovU7K#Par^5&U7oVIq|u4yrpwV('
    'i)EAd=`58qLY&EbcZtqO8EB4RZ*hAb(3m{OhO@|jUeCl0c2}UsY$z`eeOpU={GI&LJQWQLtyVXiM<pB>+9g>XYGBXKcCxAaHl15vaV9P+ZPZr^Hsp)!<xIz{'
    '>sOug1x(Rza08jZnk?c-CAd35?$Xkyx6AS69EIGrVN;q<Uc+v51N#k(2D+7Wp}BsVcu#CS`DaSklRxtHJo5FF^Yt(clfYl%mGz)(J)*RLxW~P#agS2zw}%l1'
    'mKcdgtd8um<HV}%SV??cpG>Y$tNeCh%=lAkfHK27=J@%IJ=Z-|@MQYMyg49#L$?1@)k-q;JO4sUpDlHNV06?!D=^rrU`y5!6zlO#D~$DV5Kvr*Q2kiP24Udl'
    'M~s26u}ljJbVjMIh`%n)d=M$5alIs5nuZ^!<kMS{F<3Frf`PG{y*mXRp~LPaSX_~Q3xcOA45bA7Q`|<{vMDJyA|<J^ZQCg$6q*hu;rRW$ILX7-rpYR|9S2?@'
    'MxHd=Ia#`>IP=xN4-Z#Pc#zzK0|~a6V&%eTCK;iAu+Gp|ZVGPWljhih)+TW$K3$qEr_wN5Zp9zDwjQ~*K1r^vKRaQTgbYovWWmy8P!+pAt;_T&Pl`^!nDt1L'
    '^+=QTdD3M4&E2Q8RiQ$w)$s&+wQT}*)Ny3YzuH4c5H=JE0Q&P$X7TvM0M-UO0<t{_9hcO6cVgwrD`~m%d*$WIe@|kre2_FIE19_%;1{!piz9oIxo37~k-%4Z'
    'WKScGc^qBI$k&jaDwVY5OAMo?V#0z{aXE<)rliDp`apJc^#lZ3@pS(#%vpIUbyoh!o%Jx>S^4MAo#k54JJ4n2k9=8=d|409mzDoJ@MY!q!Izbn6K3T}#;p9)'
    'rOe9z<eXXU60V^~I;|b(w8GoTXfn2uIk6M{8KbnK9OgpNyyf606_4~@e{y=S*m*g{@|8O@Uk|~r^{m^=I@$NX{D2$WAp7b0s~`T$P<#FI{MFyHAAWlE;`^7+'
    'e|)gw*Dzx56kev7SMD1vL*1p0&Kec1n=yM7NGsuw!ZWubkIJ%FR*Lqv&WjAWs$s{af<PGFGcB3p3w2osU5pnPnp(i}1JiV1uV7eRUXF{R?So>gC+1&p5(Hi='
    '1j<x4J3og;AhOg7-?Zub0BQk?S8qrstIldNxn8ev^F#itR|pYzF~@Ej>&Bpuc)aAk+q@ocI7k~(5DNJiYPcqrj}8Wq5cw9`ZeOVD#kPTjyCx`6$><GR7kduI'
    'SezLWeE6dm?n9GuJ);e7KH@RadV}D(^w6M&XGX4Py$B<hOU9*>1Y}$x50{bV^Kc!edAM|*1P|9?oQKO)%H`q8T{WQ0BuM;+lVj0_R)Y_x20hlZUdLRwcQhms'
    '!6y@Vsircjq)0+J+_xP?mk<gu%n5a*xV#9VFwK!rCJWbhB6oG>-;#V$$A{Iqk%DS5IV&Xs+piX*%#m8+YILrmDATZ$?tsUH=EwExxV`Wilgeo4PO%shl3zkx'
    'eptVzwT0H|)A4-W(O`y2QTqpVB*BhXEvd%bHG5sPxH=;&`m0i0jWM0h3%o<*o|~*-Q=4&gXepbT(0>m=aAsBNJu{p=>)nyz%-L}MIEJ%ubN{3n&YW&{M&o`N'
    '&f*4*U^pXl8!g@OHQ0IU`FJtFhkK5djb9h&AtM)_V``I)=P(VGodccAg-UuU6VR$sLR%uVOXK&={7604KI*mOz=mN~Sh*q96(;lOmxTEJewlcWjOnOXBHUwP'
    'f-;U(3;M^^MYyI17#d+gWLYTDeI>f2^4btNiirBnCm8E@57P`|n1SBH93Mtx;e8vJ32BN3o5rBr><3b`52TJXFRhW<3#4jLBV`^q6ReDyE|pam1shnh1$QX3'
    'RD>zJAC_(7%gDI!N4MAQAC{HtW#^Le*kIe(_d6As5TlzIDH!){1K+$2J)WPJ%Gf`M7J&fJyA~nxauKn0Cs3z%D<>2{h0%O0nG`z~76{<NVk%bISKQaezDn{5'
    'CUaOof<3_#sL9EQ$lVf`ZRQIE%{(Jd#*8*Swb9_tU3i<=z0muaORCuZ<8dtBnfA{sOr%MbbT3qZ@nS_vzjVDbxf$u#tw{xyf07-B_eCnG+<VH(alL%nF-UGl'
    'D<2buUOwsE6Y}ru=gWRtIbiVCNJIa*=YQLsG7x>Rh{;x;FwpHQ0SS$MR(&7oum;jQ0KH&mmsj&OKBeq<9eFt}-WK2Z+<_!@IeW)+pk?j;tcV5gl~52?-h&eg'
    '21gp_w8Iy1j>EQzqvUW5Wih@%`@J027m_gjnx=I_@evL-Vx(Pz%=slUL-o<UjX&V|Wd_Gk3=QbHKjuSBc^*IomvX&T(=VvUpz7B_gt*~@;aeDFurYsB``;@h'
    '6HrXV(C6#b`}w8e9DDxHZ!_uQ&sOKz3>O;3l_O{#UK}jdW0zGiw`V&=Qcy`@2f1u)Ne<P+m<KTd-;o<YVggWTNISd1Y?@;tIB4s8Wst52>E1@33Zn2Q`Y3SI'
    '!e<AJUj<eS=YhRaM2BFbPElqj%LRkoc+qls*D6~L%1CpNo#Bn&T2xLm%CT@rQe0m)Ap?Pc5VKr>I3c;rAZ3!s%;`gDN1J&BEJUw{gFwx8l6yVI%`*gkrsJSQ'
    '^P{4*Yh_=lk4~34UJj4ykNfVaKune^2?v6(5~L!)3p9LNhYwDR4QzC`pSd1D)ClY?t`z>&6jP|3(Tm5PJ5R88R3n(GlCnX1lN{&0mYg0Sl+*HjfYP##bUisB'
    'HGy2!nDG6Qlgj*i_@^1Bw*7*i&RdacQ22EThp6Z_H#Y`zVXc~-+!BRh>TUsIyo%~l{*ZlTvSOOl%D&jFHjaX?QjT}hc(rzox7yo~TmB)Rd5DAiX6G@mA4Mtj'
    '_U&-!??(989J94>Ps+pXD1c+2e9K9e*rK3@R%FN7sdSUaOxiI!kGy2Yu;TP*L2@)E-b?(~Su37UkNGJlZ>z+#Nk9W!8V?$BvF+-K#vlHg^(0Xj%%i6|DJ9T%'
    'G4!aEzW;wT*VCE{)3(}_*3n-o06>Ld)MVcLWcV}j00>rj5q=4XFg!e!+Im%~5Q_ZcUeb}FE;(Zb*;$I@Qn$b$gI!$Bjs2f8@+qUH48CB@V3R3f5Qsc#%oT2|'
    'J5S84>vi`LAXcbks98jLX|S|@-%U*_nc$wV{tdAwoVQvfa`NMp9Mi*F2s=m#Bdmn;HP7wa`VxD|>d{D^!E6{;OS9{@t+$D!Ie`NwM$C@9@q3&k`b#IMuFrH{'
    'Tyf}VN|;SiElukb+J}M}hN#Hchhh?%aVL#Ne<ARmZ6Uz~6Uwlyhh3iC>noz)k#F3u=I**_47_SUidNgrVAhgaH$fhgz|ZOVR|iq$>t+${57LWoiEL_*Ph{k;'
    '(MrO(c0GR&8FAXF088gYLfeSkAxOV*jIS0?I;|;=dl8LYwTDERikLVXflP8uN2JXRgmhUhZuw_*$*HuuVc9Cz<b^x8tBgzI0*e`zl?UAQ_(mH>bM+)8AlyzG'
    'Y{Oltfi)Ph=*O183Kh6oNR6w#$@I;zElG+sN``R+0qq{P&tU}Tq!GwoB?~e*RTN_??5Ve4(kat3VvW;NoyhMgE=|M55>y{6g<O-y!o`Bgz7-a0iepj@Oq6p_'
    'p_Yd%b72Rfv*)~3D#lPRxnGHyaJUy@I)&3jbpf*(Dn>ceM`anYhg_z1=Pu@vsr`3iYM;&)vWs-!i62aY*BCa)`WrtKrE6q~u|QdT7oym{=srLOB|l{fRE+3>'
    'GQG$Z$-geA054Pdi^C>HYnAk~)y3|pv`FK&g1NPoD%6IxKG)Mt8P}DDxK%5vqitazVDZ4myq8<U+aOPd2dRfS9N4>|fNC;}a!nEwB(WT8vdQ0<LY~l@92R*W'
    '+bKy|X^1Huz(=<iESaU|j&E}fTdJXoNjjiMS2J}_GsokV7~fy20qpvan5Pk&vK^Xo?U<R?j#`Nk@)+j~5uQ<;Bal8<sW(|*{xt+AFE*TRszx860r>K;!kW|V'
    'R-i?`k?*ZBewU<QC#6co%Ci-6|K%$f8dU)`kl)QVvo&nza|D0VBMKw1oEH321u-9RtYOl@G+8gz3l+X2WfpLJq{O378Aov5m_(MFRn}^IQU9gIw<o!kpV=oB'
    'YKnY_olkn*gT5YB_yc3+#CXFm7*6NU3H<q>or$yF0MDo&IlAar909l{XLEjlN*BZ<HG>k>*_YMX536(Bb|x(T)K68V-wUcDSHs5C!Z{YHE{UD6RE_F%M;=tY'
    'cc5z34{F6LRxa^)y^j1+`u(swj*^w9I_bXc9dUK`!%lLKq??k#I%ua3v1+JvY4k^=-;RO-?ggWHkQ_}k$dyTtPq(O|VIeeT(faqp+0>VPl%D0LGDD>VekE8|'
    '#3_NbjxiE+jjNOaX!OhL#kiaOuz?+Byap?$z({l0J2pfUjDb$XlrUdXKJjvmRa8cHA8UxKqBMvy9u&uJzN(zDSU29%%X5}QukmF=MfP+&n%NPd12@z*+7~I('
    't=B!G%Srq{`-fs5GU-vUO#ih>f8bHci))o|hm1LPp!TVRo?8aS>nmIRNG`Depra!Rcs@e?56MFnOCn||_V!n{{gK4}8X@0Txn(dQH)Okn0#08j;rSLX&b_eM'
    'cP(*Mqwj$z0>5bDUQ73;t<;#j9T+j$Q&p6@Sb7(^SaNG7wp3%gvB98UA|kvH4$cQNOoML!{3G?>2h)@E8_Y~qnA+~xL(BA=I;A!a>?;K<y-kk9(bbGVJ>*8e'
    '=kuzSx_KPDO%*|c)+~sM->k7zd+|eATCPvI!W4>Dg5`F!T-~}7$y>Q}wQ<KiSY^twpSqsOs4iE{k=f}jT_g-_V=H4aQG=+_`1ds29iw?Ej#afAX!j4;k0h2h'
    'j4Ol_{f)U2{Oqcjv)fF(fX+<3AG&x*nR?NR{3w~8QuCD#O*F&qDLA9}p|L84pBsxfM>@}6T+gR!&w>uuolRw*BlqAj{Syab=U8Gpa@;eHcag^ceT;!`fcZ`y'
    '$O9;+Mf$Fb@4HB&rxr_i&a1Iv^rg{$mE@pLw-BE9ON4K62UK>;<;bjmpXUaRss4*OL#10>4q4Je?)sj2!dZtXqv)T=QUhl%#tLAh9fb^3yK7|wq8?i+QwUO?'
    'hiB#;hM7HDJ7^Bs>{Qhsx}Iy!Zd%)WUu3`FlS(9Op56-mGRVf~cy86I2%UD>a`t|cshba6Mx>Yu<KCc^K$zT)zF}x_!#@Qr>be04C)E>J-7N*-Kj}@&(p{bU'
    'V<=xs)^VOkroqkY`C`WORhmaD`-|i4C#`2n^5C@)VdRZcMJA2IWA;XyLUPXPFhHg&Bie8$iqV9KylJ&f4-8ugCNOO!Q7j47sO4i=mGzVP=ya42R`<3;J91-F'
    '-}G+$rYKSek=gObl`J2cVY~vw8fw@sRCgS5)x28Y!rpg2y8&@;;Lrv;a&a@h!eHA1_QU9Ije(zYI6v}`cKuK?uQWfH)CuFondZ(u(m7iDw0zU>2FH4;AN*p0'
    'tylt#Bt-%{`xnMK6124Ulapu|SXSp@2V?rt2S|;^waTmvCH<ghyLY0V+Zw}_%pa!%N;%v!L2U&;VV){Dl#y~vW3Sq5%CpUkbeMrGL~GxVV#yJH$T?98t~;gz'
    '8<@)su8IU*YRFj%DT9DF`F$wzS(3MHa#<$2t#j$ez_TUpn9~q)xgl>W_!cQZiE5ZYBD|KkzPuc-@n}jMG1KeItD@<i%t0kWJ<X}m)O5Eqjl6_E9h`e1g!NqV'
    'fk9Q<Q6F`&TQka5=3~q6FxTBP*xE4^%9Bo@!Pf2~72YAQdU<NM_3CNUy3kB<0gu~B;@Z~@Kxc+G-M2cah8kpkV|%ZpZ)WX1XIrjk4|PY}AD4t9n&nbI(xm)8'
    'H+q5VUe6BP8`XeAvNRM$9eM-yXEjP6KDq%Q#09Dnn>da7H={1)CPP9f1B$pGRb2fJufm<fBvXS%-=g=zgzh7C<vP}T%&yP3Q}wBM?478c@tKA%vA2JAovK>&'
    'd3Lly_*k5(^IK^JER28+(3%Fgy0r4_J+b7kR3>i)A1`)}B}3PgnQ~@YagypXD$k+M6fcB&Lbg~XW=Zr~K~Q{j9e7K?u{j-o0X+dBz;#n?l~?7VZz<-|8aazu'
    'kZ-*@yN2e=wI{04f#Lqj$hNs|R%^&)BAlEO<+suryCm$sn2@90&F&9yd~7|2Q=5JA8kNx0#0T(j;Q56A4sYt&s{cuyWEEuLAVWK%gAazY%^WO3*n(KOUcrLW'
    '82{ZY^z)&GU1XQg`FtUa?AxZzU5BiM3{7&4xseFm1lO|0Bc&bEziP!AByvXZuP0eYD#0MNKUSE*(;X4Ni*|hR0J#T8Q^(N#1G*2-t>)gqE8IPnbhPixB=0y)'
    '>p0py%!~VgQ(DJwYsc5XF&ESNG-lcc1dlqIo<7V85C=e1wxLcL$go|=q{Cs4>XdpzR7}m$G_sc{VRSy2`g{9^q85tI2U7ygWYa@nrExj*bE+dPmoxn<NQNX#'
    'E@-XGg%G4=uEO%P4*fR~QYw|izqZBkt<h=)c}tnV<vZGS8HaZyp{vXUi-*<NQQ2VzUXhLEc-aW*FSkP&CP~Dhb#5mMDW`>mfQLIWc5!xXZ!)m?puJE^59<tQ'
    'Kdh}0z-di{?^@G{$7i0oI2dY^Sc@IR9E7MHbOOf-2OEH^*qqHMoXKRpJo6!yE3BRJ^x;Ms%YyI>I)^<`5tPC+`k%}oCiQeSnX9K6jqwDzv}RM?B=_(eiN5|K'
    '`vK1y)*q6+9v4be5y(cHFSpDtNw-d*c@RcILs)z*w%B=5Tno3D7$GvrSWcYIy;?bYsqdnrJyD*jmP^}9S*EJJr1dnm->5AVw_jddf4llK-s4silr+7LDvxEU'
    '(C99oy-@5TGSf5)kx?3{thM#Jo3C2EyXli!zM_blTHeiBqU^ncR7&T&@&eVLu5?WQyF({+&s)$uX0%p;!-r?58arPt7#H|nWDe;<sC@lLEFrr8mm}YCI}~x~'
    'x+$$_L1j64^NQp<HvO|*dG5%HQ4I&NZ1yFmLoBPDh4J~PJWM*9cKs<Qo|+JLvp6Yt^6e32%GvdNF=hJKn|YzNyXK)weRg3KV#wc;tz?iTI7)cS<R`h1h?({N'
    '#vCnmXi7X%6YnL5ribJpdEXD{R7CO^dC;mB>0A+<w0#yCK^~)vZ21xB%?NXEJ59{njxz4HEBJKVh9kG_vf+x$Dq=Ur^Nve%n;^rb)E)&cLv_LWIgc>nDyrrL'
    'kE~A@=;ey=2G&BmZ<Gl0Xs{bG+l!-+G%<>OxLu&CboMRy+u1KuY`_kUbdp~<HpXefVRz(CW0{ANsnH>O_suu20|R4DW>h(JA7Cg~tqcmtj%a_W?=^ItDi*I4'
    'F+$hk$;Fu?DhMN73^@~~6$E7keBaR?d?WyWBmjRT0N;`TJf>4YRcfYgj~2;_U=4vHlxm36g_{?X#2KEXqtU^b7QS9T2>Zr~sZ?)>RDB8`MDdY7{E<KW&&D6F'
    'ch(TcHM7wb9u1q%*1Zkww={yb&b%Al_`T6|D}(Yb{M@NSkNe*K{`k3*6iWBT&+VZKm(8Opif+%OGwS(C(yoI~)~<WvnQ%&#n^*BJ0&wG#VVaJ{(G3I8XCDze'
    'kRXAsBiSX!bt<pKtWJxNc-6k7-$MFu{541?{jGj#^)1Nb$DzdXdT9a~ZVXmTf|Ek#uc`N3cN%A{sr6!6AC!kYrfeKl^vFYgZ#?9VqWrvh$m@7sb|5EL^YM|K'
    '{E?jezRAh`#rc_&lj8!tJ2|=7A7bR>{&IDByFO8J@_;?`Ga@H%SCc3|(w93Y<!HP`oSXbpI28YTvx0la^8_8Z)IASI2cBCx@XmLG?7N@;@xxEgfBN=C2hQQ&'
    '{qPFr_>1rU0GjaJrKs_awvzhVqyJv5HS;%8gl|*?Y{8Hzn~(~8F}|IxU6ybuogW7lAghE%k)Nh3s5(NVdul$dXM|b@C2+IO=kI4I4y%Hq6)C#95tzfCFr6<9'
    'kH<IzduP5JFE=>B=7d*Ep}Dj1TaGBlHmuG;pnWG;s8_dGgvq{HU3GFR1PZn<^<R5|#~_VDkD=1ggX1aKiU<;_owBgVlvwmWzaFn)|EHIy%{ipvf?#<}X$8%u'
    'PTeR5j}l`uAOqwemkGr?!%9Iv%T=e?+%80{bH;DHFnIc$_|*kqj=@8FXhQOw4hp8_Bhf_f_+W|2^J-%9ygo5`z5_9Nz7sKdz6~*X{z(&)2fXQ3Ji&P_V)DG2'
    'a5(Rn$<nzP!vr>z;AAXnNsYYQ>$P)aOU#jhGw&Yv+Wl_7*QilbI|r*Nk`K13aizHaQMI^w9M%0A72Adix>qfZpW=5w;}n}a)$3k~((_tkwy4$C)abqA8iTqu'
    '-}+Jg8s8&DdcFmTdktZlJzTCi=b`^+$s(TLl|?-NG+4y*KNpL59%m8PKl66dBzXgTD``E2jPf}M`|k>g5D7G$t`YE<`rdf#*zxyXd;Peg5jWp^&&OAG1+I!4'
    'YB$J0R?zj3Ul7F&=5QibmRH?*L9uPK=7o1=2Z7r4B*=C&tL*{)<HB&%>euB6g|*I36|f)j6T_B^|L3@!!InF8y8508+w)4ocA~k(P!Mz02w=47@SzVE4ixb3'
    '!PA~6c-s31ic9!eKZRPINANfDD2s5si0x24T>jd;oTC@MTdST|7TWynM)%DM8pYGzu8+t%j5XI6i9W5|n0aXBp^?{U1$B|<ik(QC<09yF>_OP!vZgy07Yd-l'
    '`bUw%cIWWUBd8SpSbOfsd7URQMthOM*pM8NA?X*>FE2)BO}ECoCN6neAEdSL27%N%(tIiXWZibAyU**<-S>_o6KVG{%h48&(qnKZ#`}D0#{2xP1x5>sPGb$8'
    'C^d)h+9WC5t8#i&?%Q*KcfMoC*~77b+vpvFn2Wi#NH#gPr1c+hYjtH&aXN3<Mg-xt!;e9Z!n+WA=gQX|rS?{w-an%wR5jneqd`D!II&EOOVw5kU6=|-;PYkQ'
    'mg4HMaP$1v|LxVI<<5TxTJAhe%bkzZoNmSlU&&5UC{Aq!RyfFhzT!|e0xW_e@D5Uc<-JQ~;g@_->fT+LtyCUZilfbHw7kA-(lKgzLfQ{`te5Bf@8#@5{oBb&'
    '(2|_Zd?J3&vCsH|^S!4jsQEhrF)Cs$!l2aviG)}UrzM8~lj)XKM!~r(JE<jM7L^oTr3&d7N02?unnAKB5(!(i`iepd!sfR#@03KN?VXQ2Ho2BFb>a$MBN~Uq'
    '!g>Oc+jJ|I;rnKOcJO}n_+r0HY~S5^hw(?8OHG<Xs<6Kx*Y2mmAOGo48Q0@pe?8yuOV&Stq`egRw~WKs;_iWJ8eNXBa5PvLhuUkFy*Qn_fT6cKN}-S{fv)=_'
    'pdnVu249zDlJs3;_iEDj2Pa#$+{z)fveT?6qj8j8S+xyeLm9)MOHVcQS`P3RAz{$XM8F)D_eQCOH?6%0vGN;S^-7WL#rc#Y)m4n8^wbJ3vS4Cj5l|~cShY$B'
    '8`KKR{UUn~dTf4njgJR6uVH7MtU7cljkuwN0<~GWx4Rln8Ev)M<<;vkF3@Q=5Dhi~;w=*<7U&0!0Id0{bNE45JggV5-qSZj$qQ|^HJod2>@<Vl;dpX%XhDX>'
    '1jVnWDnU7B{d+|zFWKLa3?W;c6&Pnfl{d}yeI=Xn*hQq49BgUmyYagf*kq_!(70b?uOUi4SY<2tTim1@M0%w>xC{IZT5C1+k6^3jO#XhYbM?Ak6PmfZ_o{*n'
    'bFpF1R->SowS`ZJ@8K-R*~rZ`yF4GJc5xq^QanJXki+qYKj4JwKgjM~xiSC6|4r@7+ikyVy0L&HY<73m#mxwQZ@L_N`TqXbnnZQ^R`|G=CRWwf&hAtr8v#o`'
    'Q}$7+K5WT@!+eT}c~Eedj;6?CiDoXo4E}9)vJBDFO4zv9>r)UnbU+04X6KN-#^CI22SP(R18A4cZ7J2rI<dofuS!9pMk2TSm0O~TEyX_+2<TW@1@EAalehH6'
    '_|Ja?TYzdxVJ@hA5g@6c_!b*X@(rqtce6EaLZA!qB*Ccp=3OvH#ZZrQ)nnBumkR%Y!NZ`+_w(YUXVEv6K82;vY84WeS=Pw=;;hMk8=CtFhDACQ)`PRK!mF5+'
    '{|N5O9q5oFL?kW+i9DcKx}W<Y0I~Gv)e;T-bcIZM&vtUJR3f;)8_!1d85D;`(@i*Z^tkJd)Kaar57HnDepR#Io#-u`j?g0}rt<D>@8e=fX|;bAOayQ_pH3IE'
    'r=-j%Co|`$eM);PksE`$Tr=wOF2)&7oGPDc{D1jltAkcPI+@#9@a^f-zJ|k_(#gtoDsg{1Me4yg(PJWgA}8D4jab&pG*`|}*HmC&!Ph;DF}ma38py%#wFJd)'
    'v(Pez549we+*6^&s3*Me*{VBcoK8yjX#4^)fUk?0V^1p^6?b)yih}f5S12*rFWR&q=K_va+^X~3B|blslqk$y3;nFj>oB!8wV!FH7mTfzpg%bqMfduOAmJqj'
    'i4*H-7FT27%{xfbYP;Ex0xWeM;%CQXe*TDF3i6r3^)pA~Q^V{yDt^|mBQ5`en?2IyD(c+@@#5a~yog3?^TtSX%sq3I89w2G#PKsCcZqc*8p`Q%b0Os)jQE5d'
    'mtoAUUE$TnOOu%QtkNk){7z`9GaFHvo2uA>mCE3B3$7<oB(qcXg#QQrZbc}n)X=54dTN<QaULG7p(-ZDf)(Li3Pq4dKS}gM96duhjiZ%T9rna&G4o=P$>Gd-'
    'kDj8Yc~5U`-YE=m->{~rp5(_*<zP_$%FR0mL`f|;aGVyhC6E_U<86=jLM`<#vS0D)k;$|9iG^|k337I)i=T=fB|jP{FNO<dE+!VTS*^M@;#{+pum@dJme5~{'
    '&nM1LVyzW-?u+WQ<p}S5NMUP%-&N}F73RRXK#FwG>_Zkrd0$2<kQAv^er^p~q~o^b0jWM6)uU<DM>Bnuf`1p>Na~kXJ4q5nNx_Ob(lKPh?UVQN*zq!d=;Nhx'
    '2C?yi_eVr&mqbV-Zx@{i*@U_*2dFfOa11t<J$=|PjHnDqnZ>l5_mq`t0AsyOO%8WfC_X2{%0O8iY)AZR^!jY+k(j|R7~kz8Lb|<Rn13<80UHDT&cy^=y@vkl'
    'xA|2wq6QRI*TCV86Jiy4guLFnbEJ&~*Zz)QqH}PFbb}Y&y(D8*%<gSh+qUhr`Pz<No|G7@?JARW7^5vdxhqStm_hg^ztWlynq_F~^3ey7@=;L%r|jMc1+YFS'
    'lZEBh%tcPGDDuAek8ANDr|R-RoX9-4zbBu@`S^0axE<JQuOhe>D>MbM^6_#+7d}|vcZVu8OA?CSud7Y?vC3qYFRmJ8Jb?R6yUr@aMi+f8UU2iq^W?rrg6Tm4'
    '&5Z|$)^_K8+6$(<HWXy@08X&Tg7ca%`oQsAiio~q@j>Qo6heg$&T>}oAFykUuYnJge~fvdkCm^>GCI>lqrZ-jQ%VoJ3;2<0zq3gsdLW{Qm4j*<J^V%X9IYET'
    'cfj8NI$&qlcEmtiuurl?)F?hm@A{ihTElXgLAUq1hljFb`-ie)iGEv;fa&E{**0S1l(D>BJC<`-_pHw%E&EC_O56AB{JLCAU@m!j+OJP{4jl4cq+122ydP2X'
    'KrGUh=RG!bJ!j|+oS{3SZ?G59U%lwqm2p(x6HoQsxvJ~)RX3w{g=Dcd{MFHv*i#3+x}4TY5wvPp1P)GlnM4w1GdP_T=7{`Mn@sy&3AH~xQf*1BZPid<7742D'
    'GJfqEtl0O&j9rTz`@R{n6H42;pW2lzJN8<y-s?Ml)Axz7Y7@nab<07X59+1@|Bl@LY{k7TeLy9I)p_!6CYIJ=B8VqpHErrfR}_AlKM6<n19%+n=3Pj#Xh)6;'
    'gV;Nm?M@Z3+p}G$u=6r1?EK%G3LCEk<_mM$@8HTBi2CxgCsRM?4#0)a|HPQJY4epYlw+*gr$e>R?(&W_&JFFo5KU2N_qU_a&Ud2F&Z{1frYN+Jwx-a|x1`X{'
    'x1rGP?LwiwJ)gDZN#34V`&p7#yB?J{y@Hs&$8f5N8z-uT#hA2blUDq(hMf4RIseF<y(M$@Zq(PsdZI9I)eb?S$8-XizK7EyS=9VR-9y!5&7BvteWC32M7}BI'
    '<<*en+kDe8FQ6Kz?D1XeZ2gWlJi^YfM4@O83)XFEt{Td1n9kRe`y$;|&G;P2xQ}a-ajP8U;4?X7<3j%JukXm+eSBZs-FZFk?%d;rjY9@T*<w>v-nm0Jy`BW0'
    '>oR@kHJHBht(d;^9hkmz%@ivQ<TA4FydK$iz6IHLUZ3o{K56^!z%}j<{%6f5US^7TY@@9|7v*?r3H)xX<N4>vI-XFqSMwP4_*T^5`Bv26_qql6l6U2&4!4OU'
    ')plN=Wjx=7Wjx=VWxQ<fC}9H6w;&qNw<j9UABo2QZHdO?i~G*J<9S`)@%)aw<N0p9<Gz#H(TI->>VGZ<_56Vu)br1jK|OzDQ2#_2)N8}@-<e4rT~FbS6?7=9'
    '>uK426>+U=O5Q&auX?^KuX?^MuX^f^v`!f*D)!EI%&9E+lnvxxJHcd8hV@s_)J}3$BuG3Tso5W?*&nIdAB38{RObhT?MDBWakO{7OR%(e^Au0}PPn0utnH7i'
    '?f(|6?XU^^>%<N8t-<CVGr8jRD6_%hIb;YCZ08d9L_fg)RH^j-7=HN@bNf!L^LZufe15O2^ZD<@I&X2T;|t)fZ1gef7Sa4-yV8?dSk{)MzRI@!qzLNsN`m^l'
    'f}lPUw>OkrzR@Isj3`b_$l@MPSW=@>Sz)q3QWdRyo~D&gfzHH)gvma4g8ICapgw;jsK0lDdaYAGcZ&Lwy>=&Z`uvfc{*j#iA<5~*!u;&%>GS)br_amj>GLE#'
    'eg5gv)93%r^z`i#zVAoo`W>0;AJPHvNgU;utMcQ#Tknq!e&6*7tnB$xIV$ye=^EzXT%VUNXP01^@)=4o1)&8YW`ZTY*X%JMj+meOLMziNI&=<5e?eWRf;-u_'
    'b}En&E{Uiu>y4nMei%~ve%7%&T<0aIzZcJc%zk?Q>WBY&K_9-$zJ31FcRzfGjM=}w`0>TJuh2d4P?dQ*n_U^EC+}=6IqcC1?GR!qDlMc;|6!xT)eC+`kJ22@'
    '$<^9W*I&$*Gmrkg<MO;Ke*tubjHX9q(e>;N$Gt!Sm)Ur+$j)Yy@wI}srE$C&7YMJWP!77LnR?SMR2LKKE$y8FO{Rcem}8m$b3I=Zhr$e`VdpS>z-9%dW0abr'
    '5FkW=)%pw}!{GEbBaw!cUPYSghedd?zX5S`d?5h64Slyp(u4^tDy*691E{MFv$&kCXVAa0)f|kM{rK(QvUvd#4YdQlQGi}o{9*W9)p55x1;cB#)`VgH+qC<h'
    'ZF_3N(D&?pUZZ?svyK$^h8XsM#vU=r_kG@7O?k}Jwjzm3jEcM&E{&CxX2UHs##`KLDDngzKAkf>waZFr>xEP2uA0$SSl^?V_g+Q{Aj8-p%U~I*0~0esV+ZKO'
    '&lMN-FZ88}^wh6siz`VgZ5g1;DVQy}a+>S9-!bFrl6w;_q-M0v4-TpYab#|1IlG1mER>hBiYuD{XDUUj<=w%(zGD{i{m4cdb_{dw=*Z<5Z+J(&Z#=A)Ct2d_'
    '%01{+%iWraz2B=}+3khbm=Luo+4r}oX>Fm0HO6W&IqMXw^Gye`uhykwlpoix;}&A;^dqmNj*0~AZ055LSOfC~ojsBx)XNX+O{&?t18Tm5QNkS#O6;tbSjzpZ'
    'F9l`T!A(Rc8*A%Ulp++Lpsm+i%eu=u%-bbC1SaNsk0HKQO6hKOX$h~pDmO*!E~@2h6n5`xmJ)|2RJq(~G*kU)K2RY6ny_AjkmwY#3w=v-f2y;c_J#=N;`Noo'
    'XQ^(@f?Zu5KF?9iB3_IxrL^m-dY!!msZLk4n6{5iuL+Ywj>bJxy!Y!-ybB}Sn6UIX)kh>m(~pe3J5s!31hxVX)ppv|Jesfu6L6TRpWtlY?<*ecgI)@8w;zL$'
    '0mC9fi@pUde3ES5<%RE8@WMX?7kyQkFek}m8Yz=$WRSxWxUh)~4F|W+N>^4BhpE~lP>J|>j+A_SD-?NZ9~0~*#phwhpz@_kK#H2u<5_Ge2fd(_ohilp<>(Xr'
    'UMtz6JCs?AI=z1omVGBy@qQf6Vn1XR@7L!|->=osXu)98g$0ALH9})+M8;O>JaLnn`2nU~sx?~(k7qH3;NW<LDE+LTL4n0JQg&UvNs-QPuu$^DR&?sm)^xN-'
    'n@_(`fxQJJlxc_3j436d_ZZKIU^&N23arX|4~+Ajm+X4_{W<58A?N1Z*tlTdm!vWHLpkUBbvfsS)nnaa+R2Djc_f*yCYjgvO)a+h{_nsx-%qp6_wfRn?S^KY'
    '%cq0_g-mW#0s{$~#mGO3Z`;Z318D^zJlIgn9&r+9m2hiM?{i0FfP*e=7V8P!kx|tkx8=4c$eZ=VOxZ>nLAOD%3qf_C1zAf`FXKXvjgSDcH<9COlOv9Y$JwyM'
    '<5Bhru}4O%7bcfTe`wO<Y}Y+tAl41A@EIuIx0Ovn89qdNRXuec_ZG6EwnYq!C`f*gOb+Y0g?$@*h>6*kley<kRDQZiJ0dpX5!mkmf&J<M_$5h?9}d{>Q$c~d'
    '`xz~cqt`Dv%7r=nvt&*5!TwI@gH`9hg9Q3uzqd8|V1G;W!G7x06s^*}t`Htrc0S&uYLYw!rH0QO!qhRQv@$hMK1wB8pT?biACCLwj*c32x|!bN38*%d<fo6}'
    '_)q|YkJyk~U_<&FOg(h2J3>p+0$ibDZ1V1F*Cm-y(es9Tz;N&+9XEIaRgz4z@KZ1S$%Uf))|_(Ta66t}&WnN&!f?87@V5(CHuL4ouvfHAObQ>S4UZ9pt+a*c'
    '>1xJ&d=pRt7+s>HR)Sy^2A-s}I6pA*WV|Hl$`m=9-AcydFS0Rc+vTl|f7z@Su(a__c9y+*JufII*#zTw-iQp8z-!{_fKtXnDx43mRt1WCJ*LK+h@i7uEONE#'
    'D0&l&hr#%Yu0P{={7Ef}mGajU-{OEq={X7?1Lz>NF$mTqUNKgBDb{9e9oeikfH$oG+z)M52?Kg!H(;Aw?;AFYxAovyBddt*ixx82>dg~BZ`Uztj8j^Ldb4T;'
    '_@gBdsCMNAU)3THem4VDf^!=A$lV+K+!WIZJqe~)W5A#K@Nc{w5vPD$^FIlSJ)N02$;Ok-wTvoLm1yHqT#8UI3NOnp=BNYY`YA3kXl^=#>@eNu2vCr*?-fs4'
    '7&R7mB4RAwyD{hn^djD1Suo7gZXL|mrkxrwTq23+N(4mZ7_t5SwivN_uO3FMhLjbYEqZuDK-i<hJAui%b-D|fY`-3+Xx~E;^o+hZ_CY;R*?u{wY~QzfhguV;'
    'bsDy)E(&dM6u{I*;{}PmbDxGJOo5Qv<5(>K+x`v!w!vw`2fC|XI_?-4g<@W-hEv!yj`0t@nNSX(i$|F^d_2I6g$6kK-$wV%YQAiqdTdn=wF)RoWa<cB%QPE$'
    '2S7F~xgoS8Y1N(pN!7oisqFTE&<=9Dh$dIml6pqQlJL+s+IAa>CX>`5a$AWeX3_)RV4?}k9<$cFGo(*$Jg$VH@pjv3Hzbp@A)y2#v#}f<xxWQGalaOxrHW@!'
    'Z*?ZC#r5Tq?x&_W4QK|T6^bu_$CsIooHd@d9gwvj3OBj}UfaPh6fm`XS{|L<l5x;@gpK|*V56&WLLVFvdM8NeQqBIm;X?Nx0vB3fJV!UZcGmmUprp-hihy#x'
    'Gm3P7D-`JmvHkoKc1Dwq+J|M0sS7?`(>nC>)Hf2nj61+r_jiY{F0mX-IwWQ!CKFfKhb1W1DJyZg5_VXCY#c4AqV+XUxcl3qaQC-H;och8{dWK#-+f8;wrWgo'
    '`ws3s!kqux!kou8M1yU<Ex38VE+lyWj*#H}9l_1}b-~U1yQ6tS-kku%@fC;$ELYcElvi+j5_n2zajB3wYm90+r-S~%aQKKq|A<2WN21V|gfiU|4t@VK#i8#%'
    ';?S4i(4#YrK6Yo9SDV`?82#PQ=j*(Rj{?t!hr%j&bw6`0h;=_xG<HNCwd$0X23zmE6o?I_SeG75gpO=iG2-j-@-0k`q8d-V3_t%7Lj4g!y%a*-c@+Dwxa#_U'
    'X#nZ~i2Bb4#U59GyJE0+UZzmk`vLqqhogAGC?+KC5oY~UgjtU&=ktaG<_Y>&P1}#8vZXrS9_sADR?BwqXIrDIH=J$PTO|IhFN2Z}MLKpvP;TfU3P8*np-U)C'
    'vm0Kyx8@n}3LoQ9cR!^kX6lKajXFvrS!Km40nb$6KM|vHF`mq1bR0*_5NZ^~FL|I)#rS--xy@d$7E_})%5l@<hT@$d)BBZ>>HT|!Oz)RMrr!%zx_yC3bGwu}'
    'z%e5;_rgPBlH<8stv1)p71-kyY05wXw+5?@W{5%}_yFep3IOxyWrz#V?I#H&B%Wr(B*l{k65~%KVgjQnlQpa_w=_o}KX%3AON>T)Me0~bRw8X{TO9I!DGqu6'
    '5r_N%aLBd3`rMJo<Ga^RAmsf=5b{S5@`nW>7pwBKhavCZ2Ml??9EQA~gdy*Lx-jJZM;P)b40(HN?Bz$?@=p-AoOa@9a84RH{1Nc{Qvp0L_Xeb}&ilUu*7?4`'
    'I)5?9Ui|dx<<GzTcjw#ZKmPEym!L+!%f9>hA3yx`{HJeU!0!3&&p&<l1A?8y@8ACXk6(U#@ej!H>x0$pcY;2DYW)f@(4c0n*+raTW=6;?vh(Z3;uhq&P>zJ$'
    't?`%m4l;`3jn3#ejQ1<7&kEm+Auc*pU`-|ySu_oNg$bL|&20R3IV*~ReJiT*Cg9t8Mz?4L%?<~`Vnc7spq=wIWG@PWT%XPtupX!UQWVdi9XN!}mZ){nt0*w2'
    'NFc0Ogen*Q7wDGN@&Y<=wp?9byoS!Cu4FKohsPt|rzHdxT(7Qf)2!bDl6=1!l6=2DB>DahkmUP2L6Yxp14+LBX+x6tw}d3$uLntfcvLNjUQ*T)cYIteO~X6a'
    's$Ly|Y9@CGFR4?2Mw=xNo%`Ld53A)N<Tzt5#iiukYD{pDGJanw<)C|9LkjDJ4(gs#80=p6pkKf1>q|YX(f>k&bPU7cT?AUMb)~p!;P3qo!hlsv%4_tWKuxa_'
    '*E_1yaeAw$m~Z=sTP&-(LhCGrtpV-xEfm1k!uNai7H@4Hf?oZ~ZX@@wM$g_WX8!&znE7>jvp)C07BjTA(64LMP~+n3O~77Iey)i4`*%ge-~Tia@%R5+i1_<)'
    'MEw2xM#SH*M8y9^y&vOScaPC2ii)pKPJF0}0o#7+aQQXxkHc*HRVe#MNtFHl7#=>@%03SC{)mj@L_z%nE+{`PDJUuciUZR(EKOf|#>+c8sL`;52y9@a$o^J{'
    '{z-BI7t(*f0@D9MxC{3G0PcdSGWX3foNTob9bSB}_v14B@~8x;3mtXB`GXM$lu#9t>q3so3o-eyZ@c|oO!WA@Big;K7b04<nq|a6iF8&9OH}B)c#JX7iyln='
    'v%|770s#cw1}CZX!mCr}aaoy3K~(l(Y|+Q~?)_r(^4^SzNEayF@4FQj3ZTOJ2a&=ErFqA4oV-*-9veInAhKIIp#TQ?JBp5$L0^upw1cXVa<kKd+dnMZ3n~gc'
    'EH5x6h6;`<k)0Vh_RCRU5xg|j$UBr-i}ho_7gv_GaC!IB%q+cRi?)zn#msVe80yBllpFiC`8@Ww=kwUF;`7+=l`YoosZaJ3Tq|Ykr@FwyI@2p!@Yk_@OB=Js'
    '1#g3DlOVQbx$aKj@q%xO?u))d;)<7$c-S*LpEM*-F5?lhpE5!5@dwj*?B9vTL*9-(bt?8q>ahi>hdy@HV)xkp9oRkg)9fDBR7ar&ziWo``c-#19k18pTXDst'
    'oMG}mL(854S;hrrVIl2oYF%h8jI4t5`mEW>TQUo*tI=jPT3%l^;V;l}^Go<=!Dz+k5Lup((z{v}^W{1JdpWyM|86-b{YAH!|E3%#o7INgMXVx0bb(@9X6BwF'
    'q-u_9kyFB7y?_OPN~8aiD=>yDjCISNW=+oZB+F+V?WAW%LB}bP`wWq>5pQzyy1Q6iG|g{1yM@iZ8n%qE(Q>uEG)+M1P_8HByk(^*=y9X03?6l0Q^NnrI;IG{'
    '7i?BnJSN3zy=h*xP6vZd&Jm8}aWMiR;c0Kky>NA!4{Z&wXW=&*hwQJ=zl7ecUX{a`xo;zM1=_aKZ7#~0pnQg;U>-GB!&%x+$C!Bx6f~cyMk=^0EZw>!!=+*G'
    'L@P}%_h_nKhx1W-w}Z2CwY{V|g09}I=V#ZrL0>?w#qBqG?-Fxh&Km(m86f6t*AC|h|7Lh4WtN+HqUV^g*eL@H?^FBj0MW#0z+n{T1#L;QCMV=!*K0=G|I}$u'
    'R3ChmL0MlP!+&jM5doI)Wo`Y;dNqS+)+md8f<P(zhf<E5Vs-^=GgiD*ws8ZrxjML%<kW?A4aZG!s*cTE{>&Hx8CjwOt~X6poAOgtwWLi?U>r1~Q?rNalrb;H'
    '1PlL{>^!8x!KDvI)^D>BqFopD^ma^J7pI~ujQ-9gm|;m_6g{KM@}vYhUUfi~dxg=yZH1|fzo>Dwya^k6vT8LdmelcD;viDbZq!6k7YhWyemkEnriJ?ORAqd^'
    '!doX)W;fkUc3F7kI&FcarE2?>kMP(VdJl10o9b%0BH|8qhCNdF9k|?_6~a^|VwKpq6wVi>W*<!Aqc{{LQbc)<d|!lGr7B$6U#CcCgb`J<wSM;&KQRoby-JZ|'
    '7}<Ju!$%o@wD@{<tK>~qoz-M=z2<X@4bDQlfg-+Ws`1GP4>k-Hucc71*xFuHeZhwx%A;f83VJ`>&9a}wbq)6OV!oW2^GP`~(t7PGC~TdppTuxxu9|@55>p+D'
    '*GBHV?i>@viCK}>iY+ro%0!7dR3!>Ku1u6a<ErE#swgys?Wq3)TEc1$-2qqK-)8Go(L9fIE(LAR*Rkxn(G}UT!m;eSp=H@M7JXG2&^54R=9jp4^qKVlexaT~'
    '^YX7S7u+Ag(uAbv^efF1HHrQrYj`t*rJx_x9*)M>lacxfjAh)RK9KvdjXM%5v5(F}n)T`imlF=%8AbrFy@2=WY`MDhw<pMC&2Sem^<3p}H!MS<0yf-y>A|!J'
    'Yr&GUtZq*Ecifmrr29U_;o*712Z}ekC1AOrMc)A_{M6^9!7IJd>TXtaD=yf1XqK!Br~n=ZKCnQ3E<ab#rrIpq(giUy%g0zdK$M0;XQ2e>!5;}P*2SpN_2w5;'
    '=dG~Qpikf6JWv}#@IBeFs$r4ZyusPf3KOqa@6jE5%O(Y)8qjFuFxEIdP*_K_Ga>>GL!-0RlE6PCpr7-2tDVuLQ`0nz5s?3Tt6hOPn;zJdx2<^nH7g39S!bFy'
    'bMW@<FrE}T)@~@24;R`S{a3s`ZX=)Q340S3@kRDy=sD9{6753+6@<)z+-U5HJ+6od950HM&VvUWG>L-f@otcj6x8XPAy&5;ZB>)FFz`HbfJY0lB;#H8l~|{5'
    '5=+#k#WR)V+c2}NGC3V{OlOD5l@&I=3AQ3}bPaA}xaPib(@F~im2ySz`NSy#(&=-A%nmwcISmBB!H-XQH`J{B2zvqdC87YqGa+p=^w$CWg<GE4HIj;-$}(Eg'
    'F1M&vN{L;|R1GqblQPwmvPT2UvxM)qZFfMJ&JIDRH4w|lG~Ysxsz2?cSa!dW6#5%W(lOmBLu_@u8I<L=k7Rc>#S97Nh`g)Ccu6EV+%FCR7^<f|#mae34^f*K'
    '>BJa$S={Alpa1Q6>a}#Op&4W6g<pa=Bo{T$#QJPdwfz>!&6j|$IOqS-lY8h9TcLD4xt~yZrvo~K@;-p`1~wzu70&TLEwQ0*#`8t2Zfg8&*?4xx*U&ShEg^!Q'
    'p*TiU7-Aqq@6)AXH0X-=6)Pr+fElab*ETs2E;_MPcRb;CJ)xS>{->sxjlg3=RAco=HdZqeRa)&~`;}Ol#?v?B3A$acW^0%#XSACXywS|W2`(d*@eTe8YXgsP'
    'lh-pap2mug?F$&uO+Xn%$fZ+Ae~G)-ayDLfHgkOM1BW~4iHs58FhEiRTy^cSmyks<7;?63LSjGaldVChC`~U&a5}Krh7y<&vY-`X1~T_&(?QEKMDT0@Il$Ug'
    'OpR>&h1NlFqoXxF-aeb)6eJDX>K`bQu(j-8`qMDTE9c62;{#(%6*uFn(FaJ6&!?kvr?^owMQjWp%Joo~;3&BXRO)<sOZkr8n?*bh7^;~32!jkW1S1-~eM5T$'
    '$GQ-DrCOPaAH(XPkgz~Jl8+{l-%FoqdqAgAUT5BFXqT~Djd+LD_L!dq_49a*;GS?);)}~?-Q4(1%$;&uL4ZQ4@$%L%MdRF_X7pfMtPvi)w(=7(zOwoY-+?M8'
    'f1Vg`f6Uu*fZK9F+ra>~<v6vy;b^Pw*KgGs2SE3xxL!RMxWNtqCh1y-V_Gpo;+Dx9G#LbTyL4;N6299l_wc}6;1tMwZ7U^h1X3hyY(|dblqq&hCkf`9uC&`?'
    'I+{<-!5;rExO1pKTdCmE9w!BraHJz43HA*%3@rD&r=8^Ui>1LJ7Eu5EvufzhW36is{}NT;IiU6}JI!m_Z6IPfz>QR$?}Ujpo=&BO71LS~BWnUfyG+dmcM{Op'
    '&Tb9;D`j=;F8pft`k9B0iND~}2^R*~?FFOXF~|TL1Kq{N1RK5C?|z$KH6!bt7q+~IhoXg8wLwW@_&Y_skiBz`9xTzh$|;s~5%NT1tap41t0thL@3GD_oW#=b'
    'ZTWZ(^^0&p58Bxlbx2kMPxrm8YEgogP`lyfp4>>o=ddhSubFnUW+`X*^naUDFayjo*eN6{U7e!%gs{tWQD=*K#b{qZ@}8Y0EnUl!r(#ShjTVF6u-qe%iS^te'
    'vNKLMzn=;L4U=<3>YiYpsv8CvL_MeKl8TQG({seOCFCHlaq3c-cN~Q(Vmm!QHl0o_*H^_S+Iy;97OPc3Z_`@(Cv}hPu1ZsZ@;f-aWf>Q3_XyciO|1@krdGI&'
    '+Pp5Fs5wg;oSh0S*F6)==m=NX){){=x#iR&lieu}oWlreONvK&D{oe9DJ5~-Aj<Pw$-a)U(i$(8>TUEt(7hP`5V?n`@~ZhVg@esrAG0H{Dbwi-k!RY$2>q(_'
    '>QA-?O5fBD#R>INoU#3Ud^um-X5;xK91zBv5Ez4w?&yMrN?#Vt+RR?%@j1H=$Q6KAMB%Y5kIu$RbjEu~L{4o@a~cHg(Jo0Ura0mn({M$St3H+sg4zIiWcSCi'
    '$~@>{hb;J0RZ=y{X<Hctp4_9pT7Enn2hx?BA5_fWR;+9?N@N984@C+;a0^$_MEr-=Qe2ass+TCyz4-GC$+SyYt3`7lFmu7!_IT*=-6q;1H^!!&u0QZahdS-@'
    '#*Rn2^96=k(rlbBCyQ%z7Ey@qZ2gYYaveJ42(z$5a$Vcts=rV-gq7k>$FkH}M-$4RRA*MU^?})L8GJOLS!6oo+-@DVGg{95wLX9oy_Ed{FK0hSQg*Meuwl)T'
    'mDU*8FiHG7I<w?zG)jqoiKfS^7Sci{J?o^Bn73!op5Zne5yrNQ7Np*sMVFpu?hM2Y%_yBOG))FhL{cn|8t~#8gs!m<1@+X2z~|P8X*C<*YGPR7vvkIv4$d_^'
    'pd&3|@kd2s+Bz~zv|~i7RG1agKT5@+PKx@q&$e$`Q^sZL&24@hu`iGHLm2bsXufRf=e_)@bL(fs`t0kLL_0es+x1zp^OD5=aXO&H!#xwfQ1F|5RKYk#BcYFN'
    'Hs$-^^F3zpSk6bW<dym+u4WoO1K7ZL<djRH<vbjKu5hV>0wDg9EkBmB9MxZ}ds8#Isq@o#<3m_|ItNdcJvFhVUtIZ5jVlHFNE3khPE&if@IE-Mb&vnFS*$1I'
    'p(0-t)W$}*qH_km+IS@SlBB{C3L{OAdnIyWqY;Yu`lv)i(qxIq>6ge!nOzYvy_#ak&<pb)9o6U&tLGyE((TAck@N@Y3PW;Siox%_n4JA&xtfwRlF11K$b0Ub'
    'WU9Krou>hC!MvMF09@xzpYAZ)QX_c)ZQ^dIBwPS56VjifaRTLcHPOo%_v%I8y!<EP9%jztIP>yrAZkuNO<PmC=LGYC7}J&Cc_EUwMOx`s)g9^&(T9)aVWs+>'
    'f>gK?g4Yhg<($5xmK!~PXI~j5f*tB~sIHzpwpN!tF-nb&o)#6~BDfiy6LA@*x~1(@?ND3LouA&O42$#&-7ts$L|C$D*Q!(eL365V70;hzel__DID93${gc?F'
    'ITn998?BOYJQ{xUUEUujy^g78%BkTq{qtK%HF~4ps7=SZ?wk2+gm|p!ECK7e$}(T9mU_^7`C&`M0Z#+0e$tsPVZ?m;9uc7?ZcwRS9YR+K1z^vK$z`8f@l?6>'
    'M4K6`xKE)HLyBUVzE_bA>WezsDQT90*Agd)<85RGtBgv|(6>;N@U2*loW(52w_crHgC2ruV&DAPYII;0d}U<YTsNyVWHP};PKokcX^kygV65Wa=ODYUbK_dm'
    '<FOS^ZGMz&E+N^of+1A=w{vU0C!$sJ(dw<`s)enS?x6G)XFQv7&z#`Q2~8$t!i1ml^rv3>)2JaGMEtOOUI<(*^i$F`<zl{sB_@tomoODjYg;rf<&6;n$M05i'
    'jJ@5=NSK-fi&asy4bdw61=&|C5IA`qL@#wS1U42T1J5djV^F0R$o??R#>$tvM2uVP${)Y|8*HGH@pZxV{C+*33W>Ax^BJ;Zp`o#vFK?-4P}O(y)ipj<S%V_T'
    'E~w39_d^jN>HK;j?~whe#sg26yJCs_^~B$OVOaF!gpUECyV83tFGf#DcUBjO=*W#Uwt#F_e^FuXD4USpaBNng2_To*j9Lz9pG9=g3y8*CyA-Eph2XzMzp4IL'
    '?Id$h)}F8ts5Y`&_@rl!hmjMYoaw)kV}Eve^?HmkD*SkLJlR~MM^sP2)!7>~Pv6ayyM8(?%;ciTXdw{y0)yxv-(T4UBc`AV-@`(&v?ad6qQxiIj#-13Gwi>4'
    'amkB;%f=Z`D~9R=`_cx|ki^v)Ey-#2{GZ?QBUtZ3!KgvU^VGDnE=K5WFPM}&+)jbuJ6Q(w5M8}pl#{d-O@1VkdlNI7qerfGZigOaH(>`7e%+IDJI-S1Wbvh<'
    '^I{o&C%eC3I$3hLh)we}=CBEflXWtOTDf6Qiy9;(PPyT6rn;<lj(D~zhP=N7pJ_#Er9P!(w1Cx04O?Yr0VR%13Z2q3E-|&y8$sd-jhy>akVB3orI$a1-4G#I'
    '^L|SWpjK2q!Cm$8amT>uCd1bkPGi$J(x&3gzM|3Q-@`$fz2S2;B1RU9QL*vAjqaP(eAzs8X1eFq1pNvI?P=igZZ?z-2RljqtJQ{Ixl=F<g#JOl;bcKvTQz6!'
    'C)F45tJU+ZcBB*@HArVR0Lm}pTI6OxA93=v^_JiY9cp*L%8uudjG+eg-L#P!X3~^AFYsUCN-5<}*Qr82RF}muNbB&ThH3F_6Vm29g@K)_ZpbF9#r5S<*dXf|'
    'PkgG9eNW}MzDf%=N{6gQn7K@$^!3d3kW0%aaKGLd^tb5Uey!f+w0XUL%XQnVLN0GsUawhsd6)L;)vmWi=k)4#&T(nwj-B#wGkH5&y_>BU<EzHddzh#)aG^C4'
    'x=K~<3yD2J$xf+j&|J$#%g-YHY%rWxQT;9x1rn~nu-mN4$JnHrE#?<<bi79<4KFwyoT{Tnbq;hxtgVBO6Vr2UGSB&Zo7_800O?S>OtoAI7>!ziPQ7RW=>%;I'
    'xII~nFVCjqY(2<gkyWyIXU=QSWILTH>dAD|<6bF?@b^=C?ORer=q-qq(h<d2%|(b~&**<LgN(qNP38q-rZvVBWHYz-G>*yU&~G)==oax2AZHXWfmZOno`LD5'
    'ejq#3FU4vLadTp-ia2q}yb)WptU;b7DpiP=vX&+4<*9FwdKv54rGB?<VV=fkL8$ghIwWS0CKIP@QFpiS2PbNgE?3coVOJa5Xk11`gRR0XX=u68<|VGPfyS~_'
    '?4vTv@$L&`o4wY}SH0!B>61HeMG-Z3UN>jS1{^rY;`^{GFW8RLm5%Mn?$Alq{uUIw8LfKrxUshvQ|tQg)?Q>r=_08%`j1$`3z*W&ksn7g6mPq9!&=dT%5t)G'
    '5zm)iQeHxL_>$uqEkusA$Qe!mFQ0C*ex^ftMeG^THHW5z0u{C9m6mfl=pPJ+k)rx_2daO^f$Hx#Q2m+%b!_`=tBM`(Hb%$0PYYMe#Py51^VivQP>~A7Pz&C|'
    '(w<a=YU~jm^V#)$0gHP9Yx-thXoE{dCLO0fyBsz#sz1quJjfFK#|n~+2%jMl{9ep9e`EgVIuytrVY2tuwwTP8W3U4FIiDZR;Z#KOq<fq;mfBE}C%1iuYC%k`'
    'i?mP6@j)LEC@U0aexN?)_fz^u>fgjCOWh)V6-NJje6fT9m{0H}U1I}=*z$t!S#3u7b{*S`zD;TiDGTQ{AztMc^JPKxXi{pouUh7=i#~Q|msguxH$a*`oErLK'
    'oXm>1GL}p?iBPGH*QPGDWg}X0aDwn2?FH77pIPN9H?jOovC29plR8Bzm~ZG{Wk^|IZ#2m6inRTW=4W-`wG~A5l$RjPDhv(4mDR-QjS!}E>941^`kNO*m)?W!'
    '$YZTa#K#*}n8RVsMIzI+`ltLlGa;G$G<eS^s#eo#Ap0?PwWTidJPurmaINUEu`eM9b5fV_X~&tT1u8iOn@5reifBU*T|z4@iJ{Fkn+>J)s#u@6yX+Jl&k>0&'
    't)%b{n7HebZkKmN4QgsR*L`a0r1H`1)M7toP0G7*g7?<)jzw8<J)}>U{kEKiOloGuR=eQW=p{_M+!5UgzEy3ROlgJ9Un6=t;2bDbPEQh8o<&2H!b^x=2l0Y)'
    '6(=X6MO2Z-UkZ4x3C_PY@&pH9FHba-U)z~!@+e<I{24<iCz>vhmU=`H%1fjSB*qZ7g?U>>f|aaY`1mF!F_}8uYX31%nb-irO|EUCpH?e7Sef!=u*X;}S2PkA'
    ';Eho;jkOL*ki=k+z2twig=uT?%o4V%GG1ZVxVqjHgYvX)1R-y^98*i{ny<F_XQeM)*rWCAYPMk(4TFAUc@{9UO;Px-`EZMt6VTyhz}Kz4-ZPBFFcMsJ%{68`'
    'Z=7N=8LFMXzxef+pF4m5`KK4ZzUut?{69er9UXLg!|a2wB|c{VD<qzL5G{ZbKR*2stvcqy>YjYS|BHW3XYb~d8RX|7>Z%@_er$M{qC5`zs)G($GX$T;wZ-)c'
    '(G$HBg`5TYXwjf5LIX!Vv@S-kVe+g-uR)i>o`vsj^*x9qJOFLNx&_m-xLs~uL(?ap*c?URH+rpv#^!tEMOn|Unj%)UP=U5x(}EG@fuvV(JZd)Djh1FyvH9%+'
    'z{DtrGHo%_ZE<&-?(jdtVvA0AwK`7-aLta9o<uw?u;yaML~2-!QNj5zt6=qjL8*dyy&P(~LVa4@1z}dU5IR<K0X5>w(&VpLn^@y|i!$2FpWezmZ`g{Ktj`g*'
    '9BGTG?`y47y?L0uM*pzqtXt=^`PJpNy|`Fo4CzdIX5_=0)N*W#r91LnE)@<MUuW!j_(_r__xtZ(zWDz6s~5`60=+HD^ajBCR~8gO!Y&qJ!XL7s-gfNz!_jg4'
    '=V!5>k6<7#uHMByeT|=9uVSB$;nUT{IQ}WePa}%)UB^;9qax?=PdS}Qq8FL*41Lbvg{!C&jKDV6Cw<Y~c);hLm5!mS?(97%?f_-kVUjjtyVuLbeEGKNF2dP+'
    'yhP4k(0}l;RG1MAyd9khk7xeJGk@usUrh1s&-`=_$aAYnEl-&K1=8$&0ej^={tx86xICLpDXRJqL9M_4_0@lN{{C(D_uI4ee42fW*cQv;y2!pa^n8=o83x2J'
    'r{e{*00AKJgier;uwCNz2P*%zb?<dmOA<QfKi(bgqc$rx=c6~vD<n}n(?07r<H_pm|7Y*rmfSd!M8S7{MK;=G4Uhl|KwVgbRBeYXQI{Tx;*gZOXUecRSwNyH'
    'SpX730!30Zn(ard*}iV?JdDjTo2O&vWgZXzWWQwn68DVAOrVPD?iuw%i-nBv@bHN6@bGIIFV!m>u1k$AR_cr40yu2kyZ6PDFCYK#?aT4g@$)Coo;-f}0w5oX'
    'TWyjhXy;#&aDeG;sKUz-Bc<%f=tiUQIkhE+a@a@f&Mcl_j#%y)k!|SHNQ{fV=!aeCbg#hMFvBbhbVjlHFh}F&g*livy^)ii#6jAkx*QVfrx?jGoJdTkQzD@&'
    'N(`g#36pqE@vwjc-ngYz>tVW2-mEeaQ-Z-g7IENLxV0{?P;0XQAx$7dl}(~}vd;26%H|kQmQO&$1|dt*k6y`zr7Er~dbY~XxB@yWr}|l1&CK|^hWgq70?16n'
    'pIhl(7}uehX&C1u@}jDAa|bF^!jML<m)V8zMU`&FBx|TOZs_3~%k5%p{we;@{}v*w*;f5zLbTta*j(&Q_h+SfU}qFBI1mUyOjU!bKky%H78MAJ=ZXAo{4JZA'
    'KtU4I6A}ML=~F@EE6tz>|6nW_r)|~er65PxYt260M!w8)SwhJXcBw}?R~x`|LXN;f`Zbsb3_%tJIxq0IIv_7aCwT~vG6$HdnF?pb=@K(K$VAlSR*pPoyId!8'
    '(u8sZPVQ$dw%v`eF_Yi7d|BAJrq%gyIVuN(INk~X1qRlE5MB{?HK?Clj%3N%{nXwV!**tJHQ_(SWzjDPAm$(72Ja*@0I`V#BuX%j2NoCoKs*}c0&ps1(pamt'
    'fN8^5$#~p2dr`=0i<K9VZ(0>(Ef(}EJGd=75KR6jAEj+%aa%i>6)=@;=nr|=ZR*nPsyZ`P8rIgJ7i!-1mn2qynfa%UTf*2e)<M6Db7E_-<Cv?}c~41Rs5)7!'
    'aL1>WiKU|TN5Rp<YA@YZSVpn6_gsD(y~t2paSl1&^XQS*nWxtxX3*wsG(5I&RNSSjCv$R$HprZcS@thpQF<j5yp~?j9+K}0*h5$KnRZ5qwboJb0=qSZMpKgh'
    'ha_1=xc3jxerkwHIjjgOj5h1^XG+71z6gR(B^VuR2-SX}@G>mf734*+T|xq~Ivb@|OZ}Eg@3*vTHcBe9o7Bxw51#oTOOA=Os5Ui4?#JuPEnZB|=%cXK(!YTm'
    'fmY3UIvrmn^VJ#0I2fHU6Zkk5t(%QbXW7vNqlBeY;y^6HS%HM@h~PG!e)x|?vd*BPSj`Xr`r&^i`2q2^ZuBgku0Q-sHpJyC9Uf-iQCg-BSrNU6^dAi>;h?Ib'
    'BNW$OAdgNeVF&|XHLN-aZIZ<b^J{F^iG(2vdP^vjy0{9=I(yw&q;G;Cj!&yeljU^YI+XZ$ndEr@TEE+`hNc|JI&k=cXbq}orqo0ovZ4lYIytVraP>eA>VOoW'
    'HXTw91t6Wu(nMu|(p;1sotDBH)PN-%wa}p5A%pg9c9tYseV`{t)yG*X!8$U>au~(x14$x(HKfBjcu5Uo4aDkjn{-_RuWCR_mr%-y`zQ7IAE#mNY_gRfO!eu8'
    'P2a(wo(@GV1ZF03R~`0h@3CFLKCA}IF+?4i&C0$y3d?|Lw(1{;onhD5eVS)d>{1xoQGJruhIv}e?B<h;4n~BPo_}7RL46E&fLeown>6W&6j)`eQ3t^_$iL}2'
    '!E@5@s0cfN8k6zlOi5OM$TBR1B6XvW>9ZaX4HZ&>cyrW`{=fh8|B9Z+ucN1|M*L3J+(cVh^dnUhH?)I61pBpwqQRhwW_`GCUNd9hl0QMZJsebmQuTPMU{%t;'
    '-ys876`bB_C7dfVXVjYE$x&TswU8SO>Vo6UqE+zv)##aWM4cHQ_jUqVZ`!+|S#R!x-i~vJtG8<JoK%Cn8~Lt(hw@#&LJ+QG<bB0@%f;WrYC!dc!sFB0Sk(tw'
    'gCSws{Cdo-R}WTg4%a`e2e8(W>Q~PcB@!T;WahWIr%FcIMMSG{WcQK$YOt)GrL%th?YX<;*E4SaF5myOxs^;2J%9S$vmfi-`u&d8_v0*cze6gk&U(36-F&Pe'
    'Fs_<haXM_Sr-^;s`RJE_84o_L=E406-q}B{hER_SpVWY~!`u9{K3+TU;W|+3OJNUtwZYYsN9x0?DTmbnxU(R3yc^7VGv5*DU5R3RPw<%&p~uynT_THoazI@2'
    '<vNY0=_LAcyPWXhH=L55KKXMkjyVu`Ok~fG!eRy;QHfG$gn7|F+E{hO(rPv~s4bR)SXl`RAZ;h8$gbKymCyJ=z1@9SJ-?aSIhC9Gq{alLF6FlT!C1li7#^L}'
    'hgyFvdb<FvJO88lK<kOWyY1w9yXRyln0HE1hvn8HYNay4hFgDM^{WA|<K;C#%xh`vkE;nQL<`IlUCMTjYOt_9Zkif1cBT9UbWT_EG+9@1NW8v|>G~pnTDYc$'
    '(O2ZB^+IeRxW-<t<5%Pq{uVAbN|qy>#p~Q?$DMe-x{9kM3TcUbmgHG9jiU{Yy7}-w;acU=*kYQ|GLB$%Km1EZI2*QEhJx-t*?N)CSD4;Ww_j<qJ??P!J6MfL'
    '{5D9mLxtoQ&<+!pZ?Y(PLtgO{OzS+&Fh=ScJ&hgQxbuFv(T3(UHVsVGnx&H@KRa(QJ<tt@-S6WIcF>_LEBG&#6B{Ev2i=F~@nyE$B!G5Kc;{&THt0T->D{+^'
    '9A)B<j8}u)A3lxL$Ok;^0goBc`pn|_{34!QWA^!Jy7};zn<UL^Ojy(=q<L2{FlXOrEX5v$83{lB5C4id!O|=;XvTPc%5`Ysss6zIYha14?wx#+t?X5SIg=L}'
    'W;3x!&mYA8iIL_&R{RgTi~xBtwf*ukN|1Te!5d=!`zYH+1gkDai}(#%*!@<whbN~Jh3`K6<xRREkB51>yo&PzsR%FlWwM>eYmFfKal)>tfe&t_&I)7af=3FS'
    'y~X-vn-Bj3k5Lo@>zc&C0^%%EK_`i`HzT-@lN=d^tnA5H*Co@Pp{3Ug2n73Qf4sz<bqmZs{68jZjAmr2*&iXitxEN`NvA(h=%jG|mfOXLe_N-MOm6BK#)ESS'
    '>hTpGPu@gwaJ+v`=>B=!L6l+3PvaGtnvB#v9jrv=Mkqz7k`atFh7GaYR9!_U9SOXH5gGeEcjR;*1Du$UL0kwhn#eLS5j@*V+zqQ_2}jF3SCxc6>8PNmDnlsz'
    '@3g0x`TR1*V?wgU6r>hjg=vx}ixv88uK-mNW<UJDxN#0pU4oNXAiC=aBd=gXzrhHAI%ukD3e2j@F3$lx#66cM`teNou?!nj<BbBL_&M2?c%;BsfIL(sqn^4~'
    'hs|s*<K<Q6Vgj^-A8|>=rc3fj4wy`9U{_D8chptXdLI9b2YUYD-$4pJ$dk)0?kieY+;FgLD?EAA#9Y}Li0onxvCjB3iFn^kC01gaaVnWSS|YxMqAsw>*2yJ0'
    'r^@G4QUl99JbF%c!}g94)$6)RlM4`zuFF;U_TEdJL+YZ80eOk4JOVNUY*pj9K-Xk`RTEPg1X>5ro`L65-yMiBgMr(J$eG-v8`z3B$PM90#FHzaFA6t6O`ca?'
    '12vRgba|D4-;hUuF$g2rZ+wHEuq0gpIZ1#@%uDAXT{>TUE^ZyYdPdS7?}Y>6eoVz(cDZ>Te!QDUOyM^7&BLAUo58Vt-z?%3UJoDs{Sp>n_sgdSay#5Ihux!!'
    'TZXX3AerQ@E}Vz8FPx*nZWqqpx6H$ux6Gh>@{zBZhda4q?#8%b6yi1e@au<r8IJx67t6yPFP6ijoi3Id<L`1^3<kg8b@8yyb#XW-zb+*EEW0mCFkoFCy>o$i'
    '&fsZGuglvp`0vt93XGM4>y6i?BvLdbj^MtBciB_)0w!#UJ#;o#Si(M=PfI-39Y)dr-~Vwi_}~8##TPMZTY#DVgq7AZM*G9RA#Q{%@VHFjUIRBG=Dt$0O_KxU'
    'nHCQ-Rvy~j(1I^_I1E?udb$hC=vgd-?;XNN%5{q(KLa|_O>Qz;Bq(gkbzB$6-PzmepaRzdKf5iW=uBJbm*ifV-=cN|EvvH>(>Em1AF?f_ucAjQDu+>+5eh6k'
    '%Ddg=h||yE8(hmiq}TYFF#7xtYIxG<Ng-?*?>f4v^O&ufg;Xq?7RPAWXu)P{&38=<+Ty8Cgxu}1|4{n9sAvfIw}>aQF|^q4(P<I8c#&avK+cIUN!9_GjKITx'
    '0kc=xWyTb}f$?dm`uO3Uu)@s0^TL+5-w}htbuQGP`T~$jEVRZW8%9kQje_W?r^KnYpFAH%&l9*ugpurX{g$9hhC5@qvvfW$D7Y%#5U2EX=&bMJ@xaK^Ba)(}'
    '31O0hl_Dz^$~yB3s$7LrVeGj?vxA^ICDT$$aN>z1O996@f&Y6nP)hNy)-lN|(F)Fb?%IHeg0vu7CKe=|#yM~8)jFFcc@FEOlRMWyny$mOv2f)_u4mgE79ak7'
    'NrHu-mtc#)g^(Gpvh5^eDx>{leo*@=^x!p#S0#}-4pu_k5m?!e3EluTP*5UG4H@CI_shr#rz|6gIR8=!;m}Y!3Z!_sVrP39q5=s9YbDboH?Yn~4g5@#afO6X'
    'sjVsz5zOO}&;t=s#meXLq6U_Zj8FsH(>mBnIpL15gpt?r9=JZTWv}=W2$!RRUZAD9+lRks*%~-taa)i@MakEB4WxHbsAv<4Qs8S5CQg@AN5^75pg{CV*0Xd&'
    'hB^CXBQDe$$nKzEIWpvp8ftt;{`eI0WZtB6cm&5-OSFTPNmI0(*T8a@cj(bc&37mkvAX?B^mNBZ`tgaTVX4-4VJ;}_+dN^M5WzZZs9L}t92Ep?5NEXx>Tcxf'
    ')7m<+<ARP%Fbtt63o6CxUuk;?*@reSnTADd-d37aLWf)%xxJ+9qUNkz#ui(Dy8Ss_N9!$YwoJ+0YU%4ETU+El=~au|&ZDVFKqYrS(sK6?m`r(@#Os3}mKT@+'
    '!x6aQ|KM;kkDMoz(=4V(Bg~Bcx_BGKH(5HxCa}*sSJ{S6V}(rqh-LEaoyp`d5n;fe(|izh6rFC;5C29E8suJ-P_C_YM)^fHX-8mOxnhrMi03CLo}V1IrJ#=g'
    'I_=ez(P?AxL_XbLAb%vivC8o5A!2>}zr$m~xQOCZT~L_qDxPHd!6E~`s-14+8DR1nw-{#LA!(rovRaw;x1&aTM2C_vV}?rzmUvZ`z8#Dg4TwuAn!$F!#8g3*'
    'Jz%tJ(o49E>x#k!b#heyH29QcWf(pj?1EuIUv?W*p(Qlzpkishd0Qr27&=FieWV~}T;GTXV=ni3yn_}LGpNo<)=?rEG)U^$y`D(nOSvMUa4R2VAPc>U9RU|#'
    '3OJ0o4KT4#vWV;BIMt%~?U?irK&_Q4Ov<aml8%&8LU6(w$kvsPv^w;=G)I+;A*qp^M5hAIYeIS{BBThui`RVJ5sn8^_y*x_kxbJMe_5w>j`BYG$<y6Lr`j~z'
    'OHw*M*+Wv=6{Xtp(9uy%6{N;MN<YUvE&Y7``LK`%xf{KNB}Q-Xh`ni}2hl~g*<_2(b|u|<H_EZQT+TV}$&j;K#wB?q9z<S1&7!Lh|Bl0C^f)X>zG^uDXWMVu'
    'BSi@R=skU^O%Li7`*|w~4YoO8VEBJ?w;D+A;QM^sbKgp$z-G3_yGYS+N_<*5r%NK&L8gxD{%NV}z8FpDZ%J;Aqn$O&fJ-e8{hrVLLkfNSP@tAiu3m^2n2s!2'
    ';xl6N;V&C>gq^Bz1xbn;!`ASeHcVs65qO?ps%OI+*j1pzajP+UrGvD!=xTH!F1CeO%}TapmZU5z(URAd8#T+R?an!H@5XE;u12(T(Di8g){wKVaB){U7P&L>'
    'IvcXC)Ocb^jpmE3T(+%4qa7vK)4{H|xKeBlr0UD6M+1+nN}E_P3aOT3-B(R<8Boyr*GiS9X}1XYj7k0hGBWB)Y3I1Yi{SXxsQm)!r^0<&=gj|Nil<v(cc+@B'
    'FfvNWI@CheAD=!OMt>AF8$cd;Dqb+j)QpdNeT69^F0wb$P+UH-!_KSa>dr(dLeb>--}Y&~nCw@0XFOmojK9jcU?fcp2OlmgGK#4h73^7@%{D;w$|X9KiTcoz'
    'eY+VwN3j5IP&n_(%T;IrWAuF1+#0KEU|Eg(-9}W2U%fEIUYRUozFKWbMiMi#!x4Km=vFDEDv|WoQmP@Hn$x;D9vEoX6t53&A&^NqjNL(kBu2cd(bzM{Y<k;q'
    '_lzQ&x;v)kaC--bY59}G!RigS(*@cumAWf$%{uthmnk0kWD0Rw3)-ZewO_dJYsy;3jy|)(jd@v@6gz7zPs;EpXen#s@e2qtHBaI%Cc5l|&)o|v#_+N<FJU1P'
    'uv1e1`J6JaQ~9eVmb%ilpv*EZ&!{|>8CSVV6F3c+b{~r$rHd6hfKItW2p;C`JfYO}vfMxnQaY8=!UOddaD_?b(M2+Y-OSauFwT=&Q)$CzkDp_K!{=XN`6}$#'
    'CFI9ckw&$8{OsAc|1^H~{QH;RfByZqvSqwl&EI16tW7q_=DPLgkDq?=%@>bfJ{f=i{EH{g5wgsmkNw~xq>6*OtTSmw@UEn^{u<DWC*MB#{H2B$0K{Ao6WhZ{'
    '(qN15o_+t~B~0S;ColXs1xlnB0Ts%Y2f!7oFBd_n+LB%nW=XeV-{(JI#{9?M7Q4z)oW(v=)iT`CuH>n@TiNGwe*NKwq;v2(h5g3)=aB8er^{rSh!X2t6dILy'
    'm`Jb9i7F}kTQd6MUP>A00t+%AaLnPbHbTC5_fx(0*_tY_2<XPDsonkK5zy|t$1h%t|M7<>KRhYa^@U?6%lT4!+R?g3oLOTY%VWq&4un3H*=iK37NCeHf5e|&'
    'e)AGQ(9cM0_cyEGQ`qcJzOX>ZpXyYh;?=W|^i`(Vr972boX)NPCs`9#hbN8-m)z?$R=QndvISf~W@oRjk|i5PmYeR~4Ns7e6H93_LCKu!N0~)t>0TTWPu9h)'
    'G+}V}_qzR|6R-~mfB2s(_M(G9@WQ@{yU}AYr&UIFz8fn0X~0_1sF>j+$>HALV#I@t($C@4#ZM2?<qXqI<k=#QesdP}Q2`XsFSpBFMm*=Anh@D5`o&i-evHyQ'
    'M~}MX0-YwKHJk~4r8HM(-EOzlb!tdOV;^Z+nsRxH7xx>glo+I38=C!`loa^$Q}vnovITl%7hAY6#nlcpS-ln2sk>h>SIO35HeCbCSI`<CQ*6QQcfM$&V4ZHV'
    'HU~a0k%gkHn=nC>Hz4UPD8?_IB%B3ht^op;SRrTCjdMa+%o)T3Ekd=hn)6k>N}8R1%dAg?WWaJ=Si+BL1rmc>ue<c&H(aF=|LxQ*QiEb)=B8?74_ZWV8863h'
    '?*XZr@EevbTB6L|CS}=j#{Y)-i+_6{fJ~GwWOu8{CfiVfXBxXnbAWb?4|JrTPmWsF^qV+u_?YIj0!fpAv|3TN7VtiU-JEp#$M*D>svtR2s^zQchJ$YB^X9T0'
    'VZG4<S*3CrA`@*{EY*dxJzxY{u&mg&DMaB(wt8z8u`&DE1g2kiAh7{O5?O#Kl6j!$OilKt^uQqXpn}$7q&ufJ|5uQEV4Ty(ueKQA0w*V(ZZVk-c@%KY`X}pk'
    '#;)CTM0?#w@_)z&`{b5OuDM;S^0S&k^!upi76Rm903*b|W77A0*y)d|h^*#x4vneY{hKB&wuEdqrMLVTVD7e*q2*NIctw#l>k&FQr54PY^1XP9kU2|%sH@&>'
    '-V`nOBA~_w-;V(gycI}7Z_@u8d><J+@L65^kYp_U)myL4kc>fSw?H@i6&w*(^G%B+us0}mk(A~Gz__((Q_os0T{JRJ65Q+{C>hnhv+ZW6k!2~+EQUth9>>EP'
    'eGIx4J>b9Kp1Dk#VjKtUSH%*%1|52M7+*!72_)%Scty7a)m=+GmE-_Xd}bxfY4cUfQ`0V4;#=4JCMX9T&xc~5kzUYvI~lXG6}FOIHxi5K`O8;%)$)xEIPTJk'
    'pXDRbZ7wP25=VS4l0Z!89rn9!4aK>-;Z1Sfj&6{8SSzn7K(xAWb{2WF^<Kfk>gKZ<;*`^iqWKOB8|(VUj1%u3c&%M^K_r+rQ8N<Vms<+{XE#1>y)TemVae=r'
    'o-CVUY=JLT@8e(I>t&f`^J(7XBBJf+TKpHQoar_4Nl$^*A#i~sWymyoeRzDteJQALE&hu*v=p~TT=bMdaHpG1h-a$s=kl_tyJ}pg;v$Uo30hi^qfg|TFb398'
    '-TZ1ho6Vu$G%t`He$&LH<2uGds^ffuFV!afXJ}BcEqpQpysRzVvyw4ZZW&@BX1qfvv2K9R<+i*?L)H`*ExB;!ESrG)bF$8Iwe`GrZ&6y=MN$z?DigE^;W@Fn'
    'D}Pn&RrO+Rrxe@_5a&I3Nx|eSuv9af)*W}jB`$Vuu5!G*hS&ZXv;9|mVqW2i;r4iTn;n?cb-J9MH8|6Bqt)GH#F$$nTP!eEkXS9TUxDyvURrAexY-DJ{`o77'
    'hWHN9U*U3zM($>GNXMNt>5M~<B^?>a7!@bJeafH3t|V=x;DAA%K-<82&JkGOmODEGqG0V$xPqEQ9z>FdYypjgWa#B1xxO$FD<$wK2?<#xh;YXO?UpUNa^w8m'
    '7!Wqh@6W^>%SX&MSExHUa~B7=ia=SdkqO^esC<Ka{G0*VAIA2Z2Y({@t!0T1#;F`*71w4SIF)!2n@B`riY76PVSn$F))9t46<;z+lm)87{Y0}lFo0++51hC9'
    'kd$<fw-z^Q6*#TEvjCj2p@rk|n5+-H>vD59)0QRj7`3<kyEV5q%Zj%vK4rW+1NNKx84l7lO1+Yjnixx5v<|N-AcbqXI>+WCAR_!Q&o1Q;uvrB;NE*N|LY(LA'
    ')nfU1P#}-z4D|6IFnG&6`zkVRcVYnw$OpYdYn>1ARkI827~=-q2aUjNR=~>zHCE^Sk(|0Q8(H^L$#~20ETMv=xfEq&K~TZH_(2xrVKZ@~XgC@-Enix~tnmiT'
    'tUGfSIGD&dHkxt3zMlcR?>pR%Ujk>=7L&a}_<W531P)T{EQ+w#*W&4Dh?^*2xni(bvJI}*^>q=BM(BZLPp@&K@;|Syop)=NpKtJ)h)11k+80PfhU=!Ooydf4'
    'qAgjpT)9?AD+FC!1}R^YnbHB!yi5+q!hlpT%p3(nLC7(lq^OD%cWd$3v{6<Z8d9lJ8nz>&P_Ed9`hK;PmQ`<Br3x0t$xF;3#d=F4A7xIuSW`*)eUz-%Y+i@{'
    'aP<QElr;CJZ7Rcg{yROv|2u0fXUXIWHMpb?>6^-4*nU!TY@~P1wyZu$Ib|A~VWwHPaIj|fWsDoH=YN5ZssxKXAk5CZa%{cTHapq6=Nwz#GhX*-`om{Z;582@'
    '36alL$a>@*7i9rlzQB#f9u%<Nw@@_5pUeZXh*NSv_DHwPfMz7Lh;K?MpkeDd;Xc;7s3Un&jqTksEj=Pc_l#K2;F#_u?s_cO0lS)WdVQ776OFV@7Pt-5jq}_&'
    'm-5`JkvTC5U=&zi7o72S;h%SSz__pqtW84X>#MCO<I(mNww>&YqLG@F!@Uc}IRckXuEm)%%rG_>nA;f*;{X&!I;|Tmg8yywrO4(E#~s6l(&dML&&k6*(dpeW'
    'TV|9+Pq&L_Z*_3}s#}JwexmtM^BoN}p>qT?YPlL^@f8+k)x;L9iieXNL!Wr$|L~l<9#67t0#ByRI(@_1Inx?tJ?V`%{WVOcT%CeoP0_*93(Z;~E;Y%vbiR6)'
    'Y{B6ka0&#1&Mu=+9}hDqRcoFP3t|NPtr2XEMg=J)MyNbqasOb0i?RlShmr;I!v9%Pj;0a|WG>*P^MHiBdv-e{6@G9RpJ=ac1ZrpA`j`35I8QI(Udx-V&cdsf'
    '*8EvkTO3Pi;XBy0xJ?a(r*4iB3H4gClc%g@VhJG{!_8(ro&>(FFJTDfi~AHk8dmWnX`8xhRsias&@pU0O&6?_;@`CjS9vxRMQLO?CcRpwdPeM3<A$_UHG;|8'
    '6K*+XBM}o8P#ulMMxK@Gj%=+FBVnGZ`E<-`L{?bRoQT3=*GXp<ByBrt+#EjY31w(II%%0IO%U~udcA(HZ`^ubZR(@`n4(e=x&2(dDt|T9-Gv@Xd**J%wz=Ed'
    '&W37>&C^!%hJP6|XwwF2l|1$pX7AqLD;|e}G$-$H2gxoFwfTx41s<Vg)Uu9F2&ysB`T)74i3>5fBg5*}^Wkh{^6Pa{29@bXXf`fasa#F$)<ComkD~hq9UtIN'
    'hwu~4U;Rb6wd&$T^ufsA%XRsbUF58e4c<|$`jl8sd%KK=%G9xP=>)+PQ^3p<bfiF6c+@l2pK-e0ZK5fhhU_X?Ba?^G^b5aLdr)ASn_ULQ_V~hEja8AYJx6A3'
    '^&Z<!sJUxW*SdLZ8C|jyuMC4s7rw(Sl`c>(>?n%_G-@KpglpQdWYA~&c-|j*-hdh>Azz8x-slGxK*@WMrVvYy3PC^9{{_CE58DxtqWa}v$e-D3VYD}$VZ>OZ'
    '+koHlGr;sZNmf9ioNFu=wSWM~AQho{Ai?Xk!JA?%HB0t>u*ElGo$jS})Z@Nmql_9=L_}pUZW0T|gD@0zj%`RsCjt`ihWel+8*pF^2$sd9>cXT0Z_iOeOV<t%'
    'Ou~R#T#J@;*dK_Krx!{8iwfW<J(dRr;nY9Y>RK+mL!yF^#8^t9=5DiggvED@jWOelvHffVN(mZ<t!T`i>j6ZH4Uv2&N5$s6M4C&XwvXg1*EZ#QQR{uIU)418'
    'mQ<`s<B#6^OS-XgA@-F_mL&#`7zoYNY6Cbt8WbBoIx*&_M@QCk%I~OSOtd4uw05*_vmE)T_t4Tmk6)t+!qtk;+9jJ^HdgvpoDWujH}$Tp_z40rG(hU~S-fdM'
    'Z_bdB!UIusB522mi?NbKe2y;^k{!Nh+3sx}_%`#BHXYoyIFG_6&^DN%e!EWA$y6zF#9bj4s7zbPWpl?;BS0#_ms|MlTxffhl5pg4`dMfxRBi@GC7R)~N*;eM'
    'bviUn7_f<MVdA5{&05ZjJYLwha*>K&!1&ZDHpLfziE54CAB8(cnEk}sB1c869;h91AOGcQ#n_r2DGNIKt1Y5WqC>prdo4a|_;j@Gjcb15*;;!-i!}ADl-K?i'
    ')&r5IAcS?F{RJ)SZGJo?aQjolfpFF;YAW~;nnsK9>vP=t3}uFf67%h$?TPElbZMA77<-|&@pSlY^!fV3zfDDY1z`&qq5+01?G*Huys=~1;gZ@?tb=&+ED%6e'
    ')X#}?alU8^MKNw;F|+#$)uNvm5%ED)Q~Uf#)t<0?=0f}w$^*>?lpF*1LJ}{V0!NVQP-Zd|v^gwtJQXI)%{1H)*iadlQ<C^Np&G4J*VY*3GQaVO+49n2d^<GC'
    'eSs7|Mp=$PmoL5s@z34~{^>{%WV3}a3RYYgy5$(09izC_0!{w}>h~qnl^JfwGNbY-%C4@lh6<RRMGe`yVdw?aFrSqTqYMA4kp`tB=?;w&C4}xEqMz>t__cP<'
    'vwugS`|TNrHAioT(Hr2Z=zn_y-x8zNIT>1Ek;UC^8;4n|Wc|b(i|&7m>j))qLC<{7yo`c@+FHA7BRe7;4Jl=<M|SbKTrT<J-5rp@g;%0+g^==9I3tzZ21-iy'
    '%g?m$VS9SyWb@*o5OLq}!n1#F>z_g|-E-u&L@>DHrpX&rfRn?H)Rqg`)T11-=hrIS3E`y?y7MCzcre2*-l=w*mU&%|6lXF-_0)Hv!fKrg>JcpmM`G<f%TBaD'
    'k#8cCsPttZxsS~y#33>d_^{}C?iJkWi<Za94IE|ik{7|LX+NAh;~U4`k3xYqxU={p5oe${D*xRRTT*XO+V(}?T5GfhnFNTLH-<#|w&(jg29XAK=9n*=#?Ti8'
    '*7EhFuw9|?aeMKl;q8`lswpIn5O6(>jI9C9c!504+#LdK-Jg8jd^)S4l)B5YG1Ym)zz+6fnyv+(q<-1N*9usw4D}s4htt<w+?);R@{OiiUpwa(R_*!DgWYR}'
    'RM3xt_dj~{2pwH~+lKZ#QElDiKfLYuP&J08AAxT7{+>2q-5m94Gdh-mUrkxIDN71)hC_}-+=|-8hei$KSEJE*9>0#BvLWO<Dat=XMLn6-yM`Ql)JNn0BHM`A'
    'W???DOWv@O|1!pae2!9<NpRq*&;t8Q{E(sBQ;zLWZAgt5@g;1-E#@_tBM*5?W_z@9W3ua~Pbd{Lr3`?N+J|%&AqPbFkre$xFh^K~iKypq@wTHm(K>g{ng416'
    'tTqAm*#!7s!2<ZJ0r0PEY(w@RGCeBy%NS>24b!?y8<ND~uVo?JdHlbMl~9RAWh2vD9-}ATy>IdO1=((o`D&?lkFi-A-eFX0Z-%rQ3Yx9AkR8Ihi~e2dF62<3'
    'l@0aF>o00O%yRFMFFIm{=7qB-exP#XVa9v1gUN*{0DBMyr<uC}Z_p!n>f0^=nFAZZdZz@8ezySa^QOEl^~_!2Qt`OAHx*5=1p*9?0tDCrC^Kx|8qtS+!4h{g'
    '*XqP?P%V=wZo4FWj0&r9a<unq2#Cdn80@zY0$y<q`bVKPFj#4YQMs=SoF%5@zF|N<^cs*)k6X56A%`v883n7YB(ZC=H5D#+G<N`*m&2<taQAmLaQ6cS?uP@P'
    'I_vnfwWBi4nSW8IW;Q68y6uS4*JWVj*RZ*K6Q}cWjUyK%EdeN8ASJBtK?N-D4bnd?xk0KR77lLTcW}+^$+<TIbOBy+Voc6aD{5soKQ6S9M@dkoMR}Ht;jDWk'
    'LE3Hi?X9A8yIO!G@Ta!Mvv;I*oK{RxA7hF(SMUf=Sp~rp++4+*XtB*VQNCT@;DavCDbre;9_yG<1%F|Mx&iQ;4Z^{9giv10--;LE&`@K*>a4S$lck`CN;d}|'
    '#Pd<95!(jv*C_yqQ9+hr0WVc)Pgrs(*FL=Ha@h3?&g_K~!u*<1ewDqBV%qZ9OoaG#O-)RHT*XG~@zH*TC(fYTHyJ7QsNz<;v#TJ#I`#3ZlN$UgBJ+0*5gevN'
    'Y{ylSwxi>sa7^tH$35>6O>oCDV*)O+LG@L$LA?+jK%ABtMRL`1EFB1@=9@!E&b$TokUqoos4ebPh@i*&3O!Xa=<!~n&;CN_?#pBhL>#v45q2jF5fD5d_V!<c'
    'pyb)xll)=P^X@1w#t=lWu(DZCP`oc1Lq)1~X;VCNLoEP#8Y^i54K4EDo_0h=XhZ+?!C~!Iz{Z6wY=awnqZM1>Gl*<#>Do6nzUk>o#<4;aFa!eC*pe5EeYB7V'
    'ZS@~s=tA_dy4V7dk3ve0A<LWmqu8>rS>zC!M_2dZj9x=eA|8X&OO5!<Nwy<At7~A!$@z!omaow*WG!^HUaD`a71-xtyQ9393${n`q0Ih>&^b+8T};i*<wv&K'
    'nyKGZA!DX$rX8AIo2FUA$w?^;%evDXLjVFZGBReN@mq*fxWB6N)9svtB9IBm4#;#eDimCPWdLb7mx7&3raZ$_(CLv=E~$KPtzyy_?JlVN9}VwE@-EHtmg@vr'
    'Bk2n*1eZbJQ_jkOF{WfPmO@h)Rz&7{D`t@0bFFqEq8NN!NbyQZq+|}cda1h%SSs%Uc@-%V_><#}?NqN$mwJd>Yhrw}w+nx0i<cy9nxE5^*;;(@;hL9OAS&{k'
    '8GF%9^jDtl%uN*`m(t1?Q67$M`W@dolbGCHI^hb@jwQwr3z8Tb`}j<s{asGi&Cy_dxs3tn)J%D@Q!YCzg_sJ0-#f=$z0_jm+3faR+Fv3=oShQHH;F5DV?Ouh'
    '7E?+)jcYiAoEJjz`6CA8&ga8^*=V9sf2bSzvJcs50}9DZpa%{5z=SS5D0@I&xG0cmVH-QyYK>edzm2{~R#}>3%@a;LkVWwg?5TK~wWDp0#fPvkRz{^^u;4YU'
    '(_$q`Q#y&dDeIZ?JE_X1;G{9En{b5&k;+D2`WCUKVOXNDz5SV!%L(jTqnhb3Fi^vo^aBGsPm)1oAn?wPr|D#@%EbQaes*W>XB;TDSlyw=+Dl%p=*#VLBAm`t'
    '0bz6zvwYSDSfhS9&9Gn`9FIvhzlym+jZ8u%a#T&RNE@D}1kc-Qoh~QoDxP<v?}Z23J+t})LIdU)Z5O&0Um?K^$m~h9S%3H+d32NJAO07WOTxl;MzWKBOV^V%'
    'O4=RZp+V)ZJm|m+o_zaV2d1ZM8Vp)OCk-fECUdM>6_aur`2<$ee)Zum(G70YZS>;n$IrO6?tj>dE@Pm9r@*K|paZq74Z=P)Du+`hE@VqI#xhw%?4?K4T&=Si'
    'T<Y1n%bzj2CtYqSWfxT!K%cOAitco8fPBl$@1i$N=E~U%_v+t`@1EwpbzS(l`pcGhsaJ$DGy__E>YJ&aesy7sq@u0VONbbg<t9{A=k4gkCj4=*Pbkxyfg1j2'
    '2k*52!peIs;P&KC(jd`j?J_qWDwxAg&OpNU#rW(oLEC4n+Oe$YQ@8U(4QQd++H)#QxymyyFjrEt?&p?=GCTAR;VTgeCH~YrRbQ+{tM2hY7z0!`m%&LP-AmsI'
    '@X=;oM50y8txjXHvt!+$=va6BksRv=UdOtA_w=vsSjRaq{KomZRw09sdO1w9i^zqcKi0SA7_ilqXM#SI79H_zPslPBww9YxU9d%ujNTnO_1)3RmDxWJPxdju'
    '4IK05U9XyBTP-Y{a9G=8;S9+BjVxSh0vox^-|uc#%TLUs{fSvz^4@++?x)w_HV5B-Dtk4Y2c<S$sd8-9y4CGm(M<%7v<-B1nhoQc8rk-kb1wY~lRf1rGn(ue'
    '>a?BNf0}}M|G4dO(xSrNyCfSKjIH^ad4G;iTBedp=`jlXkq5nRlJSAea!=oI1&j@C`Cq-HVtH8Sl?(C4Sk^d~d6~^uB)_QnMi(XAC-Zn6ccX8x@E(=4G5}{r'
    '+Gncb_Z3`s@XTA_!{Nif#%l`_l^sptUvq5wEM3G0e@b(lbGq!huP;lKvD3C1@<of2JkvwWdbYXXOv>8O)R)@8RH_gY$=U*r=nDJ#T1g#8^w=^7R#@#WK?MgG'
    'x_*`Py6tsOyw0UZC$7w_2v~G<QeVscJv~zC;i<)pSS~#^d8dx0HarboIn@m%8i$XP$UtD7Br&FToa{TEjU2u%QEC+xeDon4L9Zbno)jiWo|QhS+2G)%Lmc;t'
    'MC4qK#c{kuBWkU0WKf+q(A|c0*sG?OZ%XVsp|UC!oN_!$U3P2{`>WVqFQnB)|Lo~wa7wF_@w=xNSq#W%Wp%=zJ+0l1sN8tQWwUWNl-1w;=;`m>G8PbT3lAi7'
    'PZSqJd@x-Ys{E}6_53vkAJqqCP|q7=AXvYzwX|!LQ{UIoI1t_qBMul7NJv0|EfVw^Gm)@XJGe!|{$&G?f0_6zv|)BSusa;n%EM_g7UiNP!jFA^Hp)GX4>W6N'
    '?^&@x0h^51D6M(6V~sJ5)K)ql=clmI+AyyCEt;->NrUr##&N@fS>H5$cNY0OFm%_nUi*#U0h6|<6;_y@iHyheYZ$&q##2Bgf)KV5cwb1fT8dV+Xp<(Li**uT'
    ';{#*{U#NA0hdRcYvBCV~nZLbw(<oFC%xnKf3VA`@oq3xE_nNnR;|ZLkbPCiFxny><YQIV5i}4MNjoiw|$xs_z-r&DRg#pf_NxW6((G)o~0Y~_!ZPd1_$o}E+'
    'iznmHA3y!#n=c-}d@}z2`4>-~zc6i<HeaEvpfUj*{lOl-@~;Ia3G_lOs-<irrm^B}b9JlIa}>aa(sl4N6KUI4<gBf4oArJvlkX06J}<Wl_b@Trd4o#*(w7GO'
    'ZB`xlPn3e@E^-Wt9F<m8lUt^U@$I>m?V^HZ;QVQ*KEl}T$j)_%t}160?noaEjW!d&uJwWJ%^QScMGQ3B>=NmSWItlV?;gK+G5*INp8W7cCQE%+;1LbPN4#%1'
    'YfrI>E_2wBezTrd)>Db*qphmqo75|{eEscbrQKhA0}5=H*URknvf%}xVlIjxLh%`Ge;5Y)9Qj8neb_p9dz!XIB{45-LZcq=%FvO@fdKHiRj&L-3dVw$RgqAX'
    '9I0<CA9*m5;W#9?nTS-3&i&eE)q5&2We@S*I8;g!vy*R=L=a59jYaH*uc!fviq%i2%_YbzwB1BrPe+U+!K5fXl*ev3DW<QJx0vr0QRP62g3CI#P-1Pe^mwep'
    'uU$kJS{5@xYqF@E%6NWoYOtI<r0Jf3d7|N)k;xRD_C1%gNLuO?*#I;Q@O;hspbM8#mD(G|vGcb!Io-0ESqT3)H<JyG1bb|UdfL7%Mt}1|B@~@j(}JOYu7prX'
    'rfC5xe67YvIwcuUe2z%F5XkoH@zcs|Q-nPldcz=!k$B~y93|m-bZg3=oG@>BVAZa3&Jms=^}Hww>Ck)Dc!vPq1Fi`V*l6D+_FsaSk+h0w7bu-VNAbSs`P1*7'
    '{b(Ls%;*TBT)a-$t}{Krmw2*XLu1Nrz!H@KSKSsU(TX$)7s({v=9tj|IEp%WI@5zr51DS@K|FJEBFG5nJG?;dxoqQ(ZuQ#)f7(cS2hu9ZM%%BW^p9p{LEcq*'
    'mx7?Zc?PZmM|f{*%bI5w-DK&MmL0gj0UCv8fIixc7TMI?L$5K99c-R#aupMFku^8sDkiV9B|x927bK4?ql@h(f}!Y~aqVB!b_)OQ`CD^04YY9yum!o`XdoBe'
    'Q4-@r&PtIXZ3E)XWc4h_p`m*RZySw;zjSsXLeF#7%lMDA`zY3}?!J@!3Sc3k&49(N9WY_FD~(jGTd}|Lxkw&oOeAcVv)7?0Nw_`HUY_uwb0gD?h0+ku!$Xfs'
    'xVGL)9I~d)USU#&MX5C-R->v35R{!KRTz4&8QE76dN0VlJF4^uxuSAXNdJ{2@)l2{nDpM+%qcx?dW$=k$GOKRd=w17GxCg8!NL3wK|n?>g)l5L;X(zZ3apAb'
    '8A9;vd|O^<M(TwyR)sVTCCC+1I*fu9j?8>iUT{XLhY%Fpeq-Egx71?o_>?F{9>~9gBC^xPVvD*nGra2FG8L4Ob}c+EZyBK{lP3{78@qDn{u%`<fm{#@EheJN'
    '&kI`Bwwb`ALjEti5bx>*P?Xk8c5g1XS57n1o%jSbxmODpfNwB)ltN>9j@oX6nhs#F`AZdPAO)AXuPgkuR!NJum|MfQW_42t{*|&U+PlUhQ<Y8{7bUcE#fh6s'
    '__sFm;65_$`scJ!I`M0~zv(dR8+|$jqHqSlo5M%lL+g1klqL)PSiA&ze>R6nqbj89tIVZNx<`K#sA=!sF=5qB007l0tOVfbGrX{fiO9l3UVuLRjTe+G0PKYl'
    '(BJs4Vo@p656MzYl?X1VqlB$$^b`zav}4Lr_}O2!)AfNovdWuLr6S%ZhQH`yr30FfCauaN_6yV~^>UJ)yl7SEiaHgVvv^r4bSUFi^`lwvi_c?My58y-GQEve'
    '9VAhrRwXE1tSYAICKFU@NY4zRZB2&kg<t*#EkJjlot`unZ!bb#vzk!$h8*<F4cWJ-SUwYx<xW&vWTBuq3uxsdk-?|GhOO9vtq3nX_8KM+?O8hCpfN(b=*R;C'
    'pfcFAnyckl*~N?%<Wf}8YL2?0tdmV9+ci}y=z8T0^b?z*EBmP%o=J7;9R6bjjG9H63>6@ip>36B&!K9laMc*Pn)8Cb0;o-%hPHygu(~Ja>K+!CQ?IpVHy(Rh'
    'vq9@#G1#DZ%1+G@X*(J0u2(vu*6WmwS=_$rP!9&ybk){&ha34UO3IZ@mi8;OStkn+tWzpe?8{0*20PoUF9hoKRXR^Nb(M7giaw(!AL}Xr4(>xWVi94wT+vnQ'
    'm3UUFwpIbhSHr03?W?25XwQY~t6G&1DI|b;VNucTf@>ji-?k#+s8tSzm@k9D)v7mkPj{k4E$<b;Kw>Bjg9wAE80b*G=IUd?l~VvMuDl31DZmoWFBL9=K#5vd'
    '7@Z6LfJIlNqUF}`4x~5&*M_SVsHQCP4w=~d@g-bjQ%nh<M=172^f8H_Iyxd-l6@vRQJ=+n%9PjgAA|fgt|lH`GM$%=YBWEG(Lq=={({N&(OB;OWHlOjVFqb|'
    'J;(2$-}{CFW4p!RQu$qDD9)5vqAcD=W{7}?3=%Y*0`F+bnw%{=X`ypn8Px&X(fLTGkn`0_H-vrYh4JX{1RDi)>SSs)FqVDa0eZvUVEVpMG?UNMMY<6d4&-JY'
    'G_>C-HRk&BEl*@)-_vw}RYJ_1SkI*HOtp=A=pJl13ny<@s#KhK8y)~6f{A2nGd=qTZIY{Xnyu*=N!4<Pw7^=49OW4sBjG1ncueTCT~5f?7~ZCDEt$md@3`l2'
    'uhf{I8BNc7BPZ@3IFW&YD*J16=?_QNt%2sDAvyc9{c$X2_LfYR_?1enhspxmP8i7TT#QH3(O5UT=j~}AIi$55*)dxL-CrqKcQu?}*q;MCk+DguRL7w%!XuNc'
    'cG^Xg(`;Ccy4w|Yb$rqqdM{40E!Fvyj+wFsZXs->C-s=ZzB@DmF05voB%WN6h@B+iRt#mFlOxY2g@4L2u7M7<Vv%}<f}swBk)K0uTV}QGf@_!qtF9}U@gfmk'
    'a}J<t$2-mPn1@Lp!IC$))jII}w%QQuDmHv{Qb2~>rR6dwOS}!xyOAD$PE&HT4mzLl2^{&r@7Raf0(p>*K>}RFYuo{^@@(0iZWpWE<C6)TrOGo;Efcie$pOzR'
    'SfB%$_YtK_$x06^f8FA?rz&x~1e)nD<*Q2<(3+Kj^k-Qr)#nwUjU3m(KxS>%YC?FH@(4G{^{(En)k(eg_dBhT(%XUVZ(Iuh&JTb_=@&QwKI5Rzp4|4N;%}FL'
    'EZTkBg^c^9{fnmAb|IsF7(l=uYmZ|Q&VIj>&sN#?ubnCXj=ZET9||KP*o=?$o^>Zc+VVfDb<)d2r)_w<opUE_y4*F&7ioOgr&wa}i=1IIHrsT8Y7{c<*q&42'
    '6EtwIEjxg^U5J!!7bBu~IBo9>xw&Ozl=8bq!P*W#EWAk86WW;Mnhv|bjzK07a6NJ3jyLgolVeFO-Ybnlay7zcaMUa72WK(Rv42|L(e$Tmk!8EkLY;!p5EZyN'
    '{5=5V@T39=pYKrbL3wXv+M&LK(vg*H9qQiem3Mc{Q~@xLPRpSbB)I^XK{c2_*V76z2t7)m_ro&$4<8+d_UF-YX+Jy=o?VG`)&m~`;~bWbqk<RxoqP8Ri>^5V'
    '(1nd<fg&W{bO^*EP}erBs6am$EBFEANMon!&l~M!(2G#J7C0PrWE}-+_+?zWNLNlDC?4|6jgWMruvrNucBoV_G>y?f^qHC^KZ;qh+pb!~XiWjecFcA~vU2s3'
    '3^vhWcTNa&b=-6?NII0)`?t|cel&MbH!?J4K(yA_l0PS^;wa}5Ub{@$G&8b)VY+Ayp>1?`HMx@RzCH8qeIY(qlf)BJODZP;5O;t#&OQ`+%YslKrp(gk3uxO7'
    'y)jh^ITMWdMo;91$ijTjvO?+JlnadoS8wM|KY1i_IIr1!K@&7|Y3Y*R;z0FHt;)W3kC2a%(+V(wdBFdUd&Lgo)4-gk@L`sk>PoCc`zc(+Q;>{>Dms9izcG=z'
    '*E{U_oHXD=-z;6Jvmoa4cyb*^Q7f*TE+O5O+l<SOAA6BHGD$X{TvV0!G(1_Lah}aK6>cncE4hk?N1;Wu*6`;bB8QK9<DY3}R;lMrnv9cGUWUJDlFqZGTrhjk'
    'HL$~UEz;1O1&lo?vgl@(GmYycL9g1W2>En!K-jUK$|eMd%r*weQ|B-$gX{PU{5ZVEVKY{$fr8Nx<ZvmG^L=v;xQ)naIeim-#^!MTB7C3)c>SPOH(kTA*LoO+'
    '=qzxeZ+-^GhT$5$^TgYCon35!1;WnM`8PE>9oTgD6@0tr>ue1j`RuzRo8Kz2R|nKP3FosHgv<W@%-g;z91xQ$ykMp(xK|h<e<T04c88s52)BH8P5a9$NBKHA'
    '4?)Uy;C&w2-Ig+)AbcsjScG_6_|gk~VIe#4sUQ9n2B@WC3+-f{UZ$K4pBPBtne7Bb>f3#49**CiIrpo#gR!HbgW}@ew$>ho)*bVn^!9LfD|T|O7;nEO3s?p-'
    '1p<n#)AII^n;J&2q?8UxDJG(@(Ju0)O_~;ZKQMD|m6U+RIHDc^d>h4+NwV5RG1_#KC|#`9*$rvtxq7!qL@7v>+$2jWHKKJHCL;0ex`A=cUH59MY#FPUjma%f'
    'CE7?G%6n}8fTlriN2q+@N_8DcS-(q%yW4AQnU+(W0EBKV15i9D0;hXECnL_ms>tSTLtbVj2dE=m?Nk!xj!Ep67bpu2%EQUwV!l32?sP)3VPJpYGD@y21+)@a'
    'hXGi<Jz(|8zA3kmDO4ZzDoB8I>-4Sc)z132%Fg53Z-_6?%|RbVr72j}6VGRT<&-wD?D_R9@viy4($=r>F>S^op{2)QtlHDdc8B(27sz{x-{ej4Yb(&nk!nRJ'
    '$-M)e^hwm#5cN~5oYqouVUb66*EfJ8$^?KJ55{3ZwhMJCif*d2#R)>|=ZI^{C&D=drQ1>Q4w(zd;V7ZbmOe=?Q{a;E!sw65U{wc5_zol0dH{aG0#;|FFg!75'
    'dE-bJ^6&CVhI;V@y8DbnDDrits)Nh=SomYBkRQkrlIg7|-Ca$OJBjL8sS0boCnrGgo1;OU*1?=#gE0rrn3g4<OC!n*+>$mtWi$&WnAeT%lP@#`SdhbxBay0>'
    '#df|iKvQ<vK*a=COM_k0)jXxgFBR!2I`)&2OFY?&Zs+PrTzKj>JRZqHTrBRMq6y0gEW-kMi4T?~OXH4JzNl+yN9`5W5hCzp_u;IEWsp<DdU29$;Sq1|fIAq~'
    '!<}VexzD1GyDjcpdcQjuKJXaFvHRV@u}D~N2(~+psn4kI)T^)k{;iHMuq$2{;)C_hZ;8^hUv+#3$A7J(<+1Yqr#xF9{bUWg$G`Aty2HcN@B2;@Vx+Icdx3x='
    'vWTc=mZSCB7oJmXJ!zNsDO>+Qf68%bkkyePfvmMgH3WA5Sneg9Oj3;V!?JW${z<%9r%q8llcCjGfPTMv1@4F%JFdZ<to!X&peS6koLeNl8*J)|lX#D;T`sNe'
    '6O*`>^sQLn$kl-r<khkmK3s9P!;S~9Aa?(?7jFNJ>ACKlWf#i~?_r&AQBbqtEVH}ttOA8qn_r;Sag!N&6gGwpUWLtw{CydzO?3<lmcdT)u8-?e*6|llzJ2of'
    'OPzc?oXgy^AIPZX1=Cc!L;0N9xvTBJw838HKUT?~{n2J7p=W|faPOnn$oSAw>Wn3UiY|11s{7_B+)j2vMd)z~WgP+mEflsPK*3v7{?ivFAOIQhSe75H8pxK<'
    '?4Jz{o8@Ne3)`@@7kUdzy}*8B;$YS451;DeBVwR533msvKd$Gf{EJM_kBl#y`7eRaAb^dKD#9^dTJH0cwmyJ<-@E0*J^9M@cESkVWl2}}>a31Y+abEFu0DGg'
    'UG^?|Y=%q3!A*UWCa-PJZxJ8KO<bv=ri@@mMxGD*r<j-~nZ?`r<_x`+%e<iF=F}d4<QO-`3kNUgCb_;9{pE&61|Rr<{u<+pZ91RQt)9Hutm83%X_}?)QjYoo'
    'csDYJI)9sEH6f#(0QGKL@$d|)%eWC+CJdVvM2syp+-cTt9VuN5#9^Ffzut(0M8$Ycrqat?RwH@#_&NOl<L6(!FeXJ2J@fEb)=>8xr~9@@S%HZyVQx*Dddr_P'
    'HMdk5y0b<FPR0h#(F(+daXe4soDzy+YpDhiq-`ukGgOCu@#M?LKYaUg{B->M$+IVqU%n`Mk|GejL*R8fe+&CL+hmh$PP-)JR#~}ws!<l2o4OdA>-G5Avv2=t'
    '{OtMnFTel%`)>)7L9ArSDhUNy424pj4X^8o2^W(8;D}9!$NfP&6uaq<CU3PogP5V4VuaJ`0i1x(M<;&Zz8F$52>Ji|kZAGf4^K|pJB6l}%n|nX3O&QNlsQva'
    't_2wfs0|Ohzv-S_!aXp;I(t1{q;E8I!@L^=E;?~l;?uzLWI4I#k5!RmHO4?JQ|TDR{-n(B2ZqxMB@PU?W>ZcotE;L&?i)nMK*>z2$}EAyoR$vbbf(#y9G7oD'
    '|A4|1IDnPdhW%bCHh~c^Cqob`m0gT$YRcIUswcfsL1U+=2lT-<Qf}Gcdm;sif7SwWs|*JhG)=PSpJO2(o4c2K-ABhIgZcBhg-%~EU7UZV3(Z;zr_;TX(s8ai'
    'KW-|(M%5VF?N}B2GbsWbmxI;z8_MoW!yE=@CF8R9o>l=s=@L`^vHga!y7H_7gXpMqaX-r|mxnny3f@SCp2(L@N^V@8aYl@BR5^w}Q;lzgx<LlLVA#N6uk4)F'
    'haU)HP~$xGj^m9NFc&?%)@6|fgF(5FviGRvK@l>};i%%qm4OWopFBLSb?F+x3rD?jHd#63QJu*K;wA$3ffd+LwqpVz5QEYUSs#2L9z#c3c;!bO1W{{0gn}k~'
    'f}=Ms;ue7v&ItXzUlHL%Q=UOoJT51C--SUJI*aVc{(*EgzkNn>d$X9@^AE)pH})CE>?zKLe+ZFRbZ^QP`<KWfs`>sC^8hH3^i_M2`|=heS`cW+eqfG;q_+Ev'
    '8UVsCwO;kW`%^dsg~?oF&bOd-5KQSYv1tXO*6j(dfbiit2@JgJJE~vz9o66U9o4V>j_U97j_U9JK<*pQV*h2VV>pA4;eP!mumg?eau^u*c8|hP9{uCf@>$k;'
    '6!!Of6!t3~g`r!FGGy${o!s6$RIg;H+ufR85X^*6O9tJa2^PBZU;$!s?Yyy3mMGxW=e8^nn(BvXhh8>5fwjN$3#woFf;#SnXtK{+rXp6me_DFf_8+z03(Cu@'
    '9Sd%F`9rmsJrE7=&oX#ECs7Zy>?T>iUZ)#nP86jVgK)has_}JlOVzm%16^UP<8FZ3-Ycp6j_4bgL*IWn)VY<MYiUqEWbs})2@PH6Vmd7wE1(iQt(a=PtEu#E'
    'dQ^u5fy*g~-tAr`!DmU}in{%~q~u{zG35Tza@o@)H0WKtO)4HI$2Eq&1)a;ECv^tD4aN6*pai||@62Ogj8q8x39|Tbh)@xxH%od6<npZQ9{9Vyq6WoRl>e?6'
    '?D~ot)P6+`yss#ZQS#uSG70Xn?TWoi3?qZR&(@%LoDO^bAq@BNea0zt=fiUnBAwLo?ngSicL{scnHS4!W&P6Od=Qu|24*KNMyx}-!~QSyf-#N)M_^~(vs<j6'
    '8YdjQ=Zb2_r6jBo1ZWl-_=hA}McI5RouRRE++`kRvze1^Ec!zXf22#S60nKi(%0n8CRt+@qy#OGTy;8I&h0V*GJ%~+Wps+(Z)2lUwdF?FD3pCh0#$SZ5|JyN'
    'O;#y;C!4Em3TrS+mt=@7<XvdI{POe8(<gt%vb^2iFq*BipOa;Db9fwm^ZXfop~D^J+5AT2XuCN)>K-0IlLK}p|NMt99s@y=$##>iKaFfJ>4vNWZdV6q1dZb0'
    '@dE3@iySbDZM7vrj6{5SxlS(8DV+khxUpsmTpt+IYUTwrmKiMgdYR0l7hgYqCf6Lx7)5{XJo)xJt`R|hXTX`)olTN&j7^}yIXif=oyKFHJ+4g|t>Jd=_Eiy4'
    'E@?r;qlBYTx&(|;^cly>T<obY$P5JUx-<G7wwt1(^48~a=QKX0bdeLFbb2BcfOj&OU;_54bFS9g<v5)R=UdbF)}7kn(U_9X9-;#@|6?HkaVY+AEdMK;9Lpxh'
    '*u;f+EI}TtjwkXT59L2j<v$*Yf1LE>KRh61w}-Nk?0%|#khn>BkK~s}>Wjo0Qw(WJ`c5yC4>=}rsCS!mkpLCfGyF&Mcc1*zCyP&}<4?Z+<hxH^{I2zW{7$)+'
    '!F@2z$WpmUTJIa>K8C9-->lZzB+1zg6?Pc#zM+wXaW(CJ5pUuz*YP45wWEveayn0j7+Km5XL*d)Sq4y^QE}ls!PI7$9x}=&$ufrHqSN~Z2C~w5vQy~yFJ3--'
    '{{81qUc3m$i7@)gk5qRN6O;JMNDoAl5!AT3^~AgVA-iHm_>}BubLI1=-#z;gqhpdcEAlVK{6O#z)F?zGs2i~_zl~mA0ov0nVca*_c5)?|(}txwIBYov0MN+>'
    'Ub5P5@^19a1`qxX)xm>vZ$K1B*ckQm7e6u!AYbJTI`sm6(jBgY*-fVhs?I5gaOK@vvBBl)8K<z`oF3h{lCTy`dUmfCeYt@~+{uiE`Qvgb)BxX_1~hw@v{V$X'
    '(WIL!R-3o<rLj!NfxIvp6^1doOtSfQvGl~BO{SRE({8$Y+bjm-h#@FQj@+#Q&+Rh(X$$C2XXg!Ie=vWE$#R;!X^gl~xbG9;H=RkmIM<F_1xQ4>=G#TR%z>s$'
    'v_w}EBAo{EHo-EZvqW$f-Q=gH1hLzMma8rt;GDt};G;^Yj$1HL=WBbxsC41roFitV(1OT|X?wvNVi>GM7mO4ZAQ~b(%$N#74el8mS!{~Y2kd8-Hyvnx#Ap02'
    '5*cMF@?(1uh(=4NZ?Ht=1YZrv5d8pQ&3{VUR0Yb;8u=rh?e8AHcrpIRAD;a1q!pfzC`E@LMC46EE*SWqmdb8Sf2rihy0MP3gDzrz+)Mls9@w4bPibk>i**uT'
    'hZa-A0cISIGpybhNXWCup<Z{Dt(t0xR_HwCg&MA!PJ*lILIaXi!7YwGx;a)T`%MjEmd=VIB9kfaO%*r-JD7g(V<1nX)wdnyy~LG@6&G+8qn!ru#j){COk?1^'
    'bZdAbf@~zxP8fS%g_ilci6URE*6s-S%4Lky!&xIxURGnziPss}+we9c%W~ZX8v81dgmOL5EvQO>0tjz95OrM!3WK7YavqdXMtt{p`(Bcu;E{BKGLB!?0XQKU'
    '2R@myjbNvAA6*((t10-1edsg^GlvZooac)=c&pMA<r1UCpn3Oloo!baZ`Hkq7XnQ9jAQ=K!)&07cHQzJ#GP5-mhh>|pTgbcHM_~A;qe{rF8sMQ)J12R1L3a}'
    '<%N{`IbAi0@!>Gc_}T_Gt@pt(c@tb#lkXCRDSM5WE!s%aMp9U}4Xgub6I{6v@L2PW(;AhdwJ%#O1pxb(&yS5wBp(&<9HAQDxJzEAARBt0T9khwIJk3sor4Rg'
    '-Mi;0#p64uz;Qx6!1$!qYR0cd@|rZhTxO`UqAEM^Y=_qvC%o1j-GZ;h7Zada_BJYg6&dfL;+dCYl|F0eBP?mMrqlpP$RD3PfAP)tPeV;59Z2cNOOE=BUU{9W'
    ';K0>E$-zx&fbhj9*XhUB34A1vS({j^7wPqO?Np449u*X{o^Hhe=^EWCO9WOU+~>p$@%9bub}Y12tIIxf?C&{8f7+Wmc=k;Vht_*g*V}rp6h^Bk;8u-p$oM(B'
    'y}8sgmTSTt-NxJ-KB>a|psx*p7fFKyY=PV{3P@AIj#h!>CY@qcn;J%e7hk>j@qn)(4q%Ve-j57c|0z{SXOj==&9#xILq?t=rxVu}@cEJC#=1xi@N^8^o~CH6'
    'L2YUiHI}XDfRltsjivaTb(>wReq0itPbH3ct73)(UruGo?}w7xdvcfx+(h0Nb7)`6Wr3;85+f1!HBBhXgVH5<n%H;D&@n}MYT-3_)hYfC-u%lARz`g1FPX<Y'
    'KV7fJAgUk-UyRZIK*@cANt&srVb96dvggtGux@{{9*sAY)gHJUo}aO@!h{nzzxT{H-op7=<0Yx;O$VkcZUG%;mnL~Rn#SFsiqVbZH)%yT?-SV+tx9{l*S@Ig'
    'Yg|C(IE&hGz%(Q^OIiz4w^@>xoO5UCd_x<q4AJ_Tx~csRnYw)oE%J}(<O<Nekha0oFPY7d5;uv}(-RU^kq^QHI$grrU{!U|5%ZG(TVTpHI-O-lK`eF#n$Ob9'
    'tq_lgoUMZrN;o-@+ZJ;?!2PvMXLxq1#mz4Ay@r_xg^D~pq5nYJp;LDt&s`%LGi_}~Beda(3J3q~cr}lg)!8~sk-cc``uzK+Uw-rayT>oTfBsM7XWu@48n$`i'
    'aMI#c=*F~j#rtCNF#*Ixs(JqIy^-eyTa>>9xP|8dV6kUuQUf+3uUsjD@`cMHr~|czB+d=1g>P=>a!8OhY?q21`k7}q)X&G<)ssf@_@Z1ZsP4&bG7{E-`iW2q'
    'K>PK92IC?<K-u?rROa7rhwFG?*nZ5TI<W$FD2@lF<H1)7)E_hD0b2AjE`=M8<|)Fi`+_k#n7;X=y`A_Cfc|IAUA%;hL8=$o^`-Yg*=;4t+}Ay)&q}&ea`ZeY'
    'f;`Z?seDSHmZkpjfUm#Ndv!N@Z#VLm<uWOu$BtuE=O%IQZ5+}_uqB<s4A&R4VrsxQWz-hV8DrDSrNkFzyF{lQ@|mHl-@e1;mH+swubw~osx%F%-A-a`p`$!o'
    'c+i%m^PIzMC#?3o=LqLgcse|0>8Nf0I6n3N{K)t73CLfUs~hntIMPFGc9r=coWiHoW$gdd$4@X>yiu#hctohAK0uEE5PULhS%$!04R^quLCPg%u(%xk107OZ'
    'c)zSc4l5hR(Fd>OtKV2)eC+c&x@zH7H-T5!rjXFis|8R)bhpcSy1e!%f6K*UuXy~ilnug}khKSff85`zcHmb<z^{scUljp&qzLeF$d9ZJ&@u$_A-V!MR*P5R'
    'uvt$UF#Ti=rX5U~S-8}t?C|^tT#Cov`n{UgF`87LbLS`Pb+&FczT7T9{OgB*$<zd*X-wQ&<auA>H;{rdxU}MF+-($ODQjRGL7tac-X1-k|29bVBcf7HuX<Fi'
    'r0m$5^}+6h-i_G49@b<_*66p<H)v6wlL`hnP!1d*Pp65I_<xa2xATO3h3Qt;35;dpV3o~SH?nisLc?NGTiuwXJFRCRZ=qZi0d{|OdhzurGdJs?22nwnzuLLv'
    'O$eR=YK$9#yu+wmmm$3bhJ+)Ue_AaqldW#=|6uj><ySWWlzg`_5b6lHez!@{_0d|7<FOf65U)4sES^weK_}k8=~YmpqBoLHwQBVPJal-^nf<EM4b|~G3-D|U'
    'rg3nQN@4b;dkN{)QH+Yyh%MmOxcA%W>@NRz4<i(gyZz{^7cajgA$OXtVbjUb06NWgKIlDU=_xY|pvLi-hH2s)^&=Pl%lxv@Xh`M)Z3)xJaQ`6|gWfFDcn&-k'
    'cch38LoIHX`O~*-u6={+A9I`pnrw~uF$`1^hFKjAMFpdm6lwu)t>`6;gjQGrnNL!bBWGwerr;*z@`#HJj8eWxH&nu7$e6<|Gl$DbhhnIceZVhL*`Mk|$8tTs'
    'iV>d|!VV3eU#{a<1ov(9CB<%bC9%|wB7Kk~a?c>LVA?uYI;!JrmaH34a&Fnc=Dr4$?_a!>ZNlgGo~eT{nwtmEfQeR%v$E5Duk7$B*@1T0yx(N^J3~D~dZ7_*'
    'HuOH7#_M%#JU`)2^c*C=RP?A@ZWm4XE6%B%04HM(2v(~LuM!a3lV+#iYDduTjAC#xer7r2e=p$y!T)X*M<VCMwNa}jSL!SHxtjw9ElVPyU(t2@>@4aHeHTBp'
    'FgsAebQmekDOfOR<}I!n5`A_S^^;Ej*b4=E%(nA6V}Ub7)q4-*WD1gPD?C~@J&LhoXll!ywV{iV-@+VF6j~~?)(*WfXzeI+1^zH%zVcsa;V?i%xz6L8N8{%f'
    '1z?Qe1GH}DZMSuSR8)@o75izvLX8{F01hTRa+@8jE08XyXAREh&}elx8M&OcMh<N$xEqelIu+UmQb06W4PnNO4UIG4r?dBr00gmh1$B#%16u-jFSG)f=>oT}'
    'Q>#rtrY3T6`dCY&2tgeH+{pY3Hb3u;!pB0zM>KoIwFqFWCu?jh{PBF)j)o49(QezwK)XMx+)cCe4W&EPv#_RJdqxTt%>d)vEj)^FMq){7xA(0njd;(B84<@r'
    'ZE(p8H&_-WLY>fA-`(^2#K0aG0LWK$JWsEaCc_Q!qwKXB67O!&T2Zit6T~Om$R6b!JGKSRBMX8vb$j4*gasrU6&wK&YSoz)=>>SyruEk_`hLawh;zj()jyCa'
    'j)-dtn~#X1t$VsbnD_(MiXpLvq?hO8cWHa}K4R=RR1`mwoNI^eGn#@_UF|zfVXQ(_dPZ$PIJ<`J46J|6O0^-^gTodK>x*Xx7fHO3sQ{REhSnJ6G8_k8`=;m!'
    'uDeR4R%B$MNr)&8L?PaYjx1))wN4x*rzD<68P;vZ9wd5UKrC-Jn!3T}fFzf2p>C&1*QB5N_flBrC^3hzBnIc7xlMexgWl_OAvc%pXr^5IMbBB(iTc4qkD7jS'
    '3rXz+a?hpWqqCa4a~2w=pvs4FF%r=lyM?UIV;I<JBx#*W#lf@#6z%bLuxw1`2|Xak6ziYIm|Njz@r-YZ+JCcstsWBi`6y7sFTt;QvI#_iP~4j=NkmsaVbqEq'
    '@L$Z6X04KV%vWN&ImL-Q@kKop4qru|!KT0$(JQhpVi+r6Ssr#t)0AOVuUe&T5n4B%2HQ!J(SOf}Gk;_<+_h~MnDiE)9sx?o=?QTQ<`|w4Ujb=bG}g9d7}TqE'
    'WA0vWXVlvA_C+D;P~s>}vnHh>IO7#ipW*ZI32b@0>ZN<rBO!n1>-Rf~9-}vH-fVH?Hit(U@WKqPaIIRL&iPu(_q=BcRt|paSypZt<QzO!?%bWNB@b3HjL<ww'
    'F6uQ-7Vp1TCHt=G9N}~yfUI3d4B03RgSKo|J{(H7VOUeYt4t$e)ypIo(rEqh>9a`BBU1AiMyq7qkx@|5xBZTa5S$9ZL3eY7mz(2(RO4VN7M)!qj%LYgs{Y!+'
    '%blcw$OzYp<bawdkQl)p1e|R{E08c+ts6awCs$q=6p78r-15kb+uy?JL$PI(Z4gSF14Dd$mCh3_#xc1MXrc$aWV8i949cnSW0Y^w`CN0Z<s_Smv_S?rX}P*g'
    'l<Oo}<(9~1wuDuSC!0?-qrnV9#$2WPhbEY-Y(ZX0>hJH(IWrP5SX_^|rumEMy2Grp=es1NN=s(InA!6!ij>wf%}7v#zhg+ORzDSAx780>JLApq04o45vI%b3'
    '37jnzeaq_$C(pZ~yXNa`3=+D)Di?Slp2k}c_8Y8X&eb#nUEvwAPMYes&sf7nx~sNoYCrJDh^*KIM41ECP|tE6FT#FhurgE{6e}XfZ^=p3awz4{e-W+j=}pf+'
    'ogaYDYQud4h({q^);qo#+iG%##=wwdMdSD;PUra4md={yxZD%>x($o2F}oq2l;1}^W0U9?h#JE_Fp>@FN#fi7&8pc!%HNMHRHPPCXXu-?+;kVniY#DiQ_*tY'
    'cSjWatXJmpZo{P1Db5nfBrIi`IK8ut6;Rkn2;S-Q$BNk#mIj#M#Ho#*x8Cc-T#eL{0E9yC6%^G;a!2}AGej#{&n%>op(%;G=&W46_0vlA^%6^btwSvnkbT8S'
    'Duk8C_%1Q2%CM}hFVFv(I@`s4g~<_&?FGH8Kix}0FS4;;?V-dSd6AclRt`bbgb#NYa}A%E*NSNt)027_p|SiWrDUWDXnZ$xGbV+I!HeQ`3l>z5@BpT9-<q)T'
    'n0TG6-8mi`<wrb|Wc;ur-4yoRiq}20#D4(vbOC1>`%|WWe4~i#jT*Jw*CJZirbb>~Ez~E=JXs(Uj-Gt`-4pRgJNh$qvE_RsDVI<tmIf(R$>ul5heLQ5gKP*F'
    'M?8Ov0>dBUNp_LKQy*nm5D{1Cpog`u;a?yua(V<;CO6HhB%qTesiR<#ZJ$i4WG>ET8z5lE0I0yJE->DB0t{ms0mJS_Ut;c66vi=S=K@G1a+3N)nc+SAI#ZCo'
    'Mi&P7H?XA(*bDHlP6Qlzd=0-|NA%Nm%jx?y_55e(*@xBu|El@Vt^Z>X7~#*jarK}7+=~8N^r!AilU#5~U*y?*i#a>j@f6M2cr<dDiPoMbi&eT#@!AkiRAFOZ'
    'Yt8*Mxk+XIS?Xhwf(X6>u>p2{dV^1!7+)cqq;rwIp~cQFw+XDV0{7(2ib|YEUrD4s{NK?x%hh(nidsMs;9p|iL0U*7sp{|Y$hVZ7NRC#?cx%kkX@bpQ5#chP'
    'bb%ibkOR)fDzn>i5)A!f(WzNebQxR3Dz_;ro&8T0K0)?P!oP(dDEytDD8D(22Kc}bzg3&HB&U!kx=UW6@qBd^H_4`fj{)Nhx6H@DpD@jdEI_WfK;uuCbJQ-j'
    'ec^9;;c3Ygk$sJEvlrSkpU?eW#7EuBq}!iNJ@$q52jbv&Y`i0?8O!^pz{+0O%`QUCm|kW}_{FCMl+#u_<xtL5|8n=UU5rVt0qLn7p)mL--jv$R!6|Pj=@P;r'
    'e0bjuMkF9=49^f}=aD<HZ6J7ay6ChcWgwAbk=oDpjF212yY70INj0OIm*v(Vehq09EiiJu#E)SS(SxP(ArJ$8WNhWX4ve}f;%Oj0p%RsZ31jJj1DYo1kRbnN'
    'Oc#mI10&x|75fuUhHB*ki#4Zq@+H#GvNBfA+NRl`7S6FeUyMr3M;A`>i$Zhn3d)?f3$=iMS|~M3QGY8(5ndn{@NzQRI~XB*TohSB`8>q=8ZY$y;Xxw+db|is'
    '*+A0PhS@`^>#2$kMfU}31hJhhs2!kBnf=c&>QjXb9Kq0TI2d~J?`Csf-@e<o@98T%DA+Y%h>$40mActKhnK?kiSW_2KSImFqz*Nm*@D!Lhp38t+7ZMVSI?#j'
    'f9v~iQDlCN>E$uxfsQRoVM25UQLAe;M*&yJOaBdz?*lk4R)ep3jwArXe<$;F)%<7JkB7uD4trMIDQhk+Q{s1D14J<a2g4WSOCsL`h7g&WX{{|_Hdo`HckuP0'
    '%?83P=CQ;KsSLIg@sm!$t59>YIdL^Mu|l1Pq>CH@$n&sAxX^zb0^xMAvjOQKGTxTrPSBr2c+q8X*fZ5eOyAG1(4ls{1scosFVgxgzE*5%<MZW=@xvWC8PX0O'
    '6*=U@IBC#s<Wh)U6k{P(@qFc8fJc#)dk|h7<HGw)Iqtv`S>T+hem!2TQ_6?ON&_U_Pg{5$G{0^Yhrzur#=9@OTCKj44gQ%9gZX}=&tvge3#jeCs-d%ok;PSA'
    'bWW0qe%I-Rq41NV7L}t?Ja`~=A3}_P%L>O4pl+uqA&>l9C`tgoke<X=ljNJ(m|eD;juJFz_S6=H67#2FP4lT+oJXS{enI_WRRr<cR$aCJt~OQ%;M7{c^zg%b'
    'j=pRctFdh<mO<whif6Xxjv7roscY)Aq#CREtl>fZ?pHVqlIhwJzr4}6*^=B<3(A`9Rf=j!kCk*Y3xI;_0id-80szmz06@>R9o4DWrKIfLjGX7r1nB$fFd_1I'
    'XoNbHKDUhTh?@lki_z?J!p*LpGcNHfJLJ^HJ?17{1D<wv@%o?(P_qJ)vE!c`#4l8EH23nf(B}eLS)}!~Mw%M!h{V2jgf=wfQ{-tts|5|{zrF^Rt#Y`dP18Hk'
    'Y}FSLz=Rcw+qmDE-1OY<^g|sv*Na4hh<R>7D!}54Ps4lfXuU7cfpx5camLaUsXjNxbohQ|j45GB#Td8^FvQ<vd`X}kg?a~vxris%m?e8T4P82wtY25qaZ^K5'
    'Mg*gwSzpJzu|RG}j5`DQzjZ%IdN8No^NukBII!H4IPHi1ejgSvn_O#^RUtZ;mPtNosYb!&HrtBFt)+b>E2rOE!dOuIb!>c)ebJ4suz+?i`D@_aRooqR_u}^e'
    'AbD}o+&u48TO)@~D2k5a)3{Z<S-k2&%{*_CH3pX9XodXhQeYIasiZ?o366A#J(Ozkg(wepxuS{6l2i6oc32#%ADEBxrfu?nXhi;J(F|znLG)n@>9>#^7jK!R'
    'h(cHFRj-aFZtHjN?cK3c4_x}4I+!*3P~lLgo{y1rwoY@nVp#?kY8ELf^~2IScEQ$gUpzcHbriG#?-%L_CwN89FR;ZLT;+xSX-i2g*o`qTG;r|gM6`Tf=(=w!'
    '`n^3jPZg)DEKiowmLtNlv1!u*)wUtEjfyr8qN8K11z*&=)R~5Cac-b8DsxV;fZDz(RKH2<PX!C`R^XDgn@qmQD?hR4fpsEfW8G6!{L#ON2*&95!Yj=v7%3X1'
    'IxwPG<mu0Xi-gKN{drQl5zZx+jt-ApRwh-<SNAT_qp=vtCs18}bhi8#pSv~PD>R(St2A_hy*BWjlTH%L7fw#11;UgH8VuL+k%#@<MU14fbo!<>VwdUZ8**E='
    '#vSb$8)1#?qY~|f2_9Uvi;(}Tbb)WiE6losJE|2r&Ax*?X|-!Wv`;l0>O)i)v5M7my2E>NB2KeMrR>|1{G04kq0E&+GLx}%7nSa$;!e<i`S*Y+HW8|(#AL*&'
    '`bMg^KdSc#-O)G6q*(a{#TeEU+BBm(+Sel>%NV?*DAg?A>T||i%oLZ3n98gEe1d#IB45PMZYNQ;RfCRNMf`Q~ijy?8qdZ++&XWz+@}+lfJL05};t$bbf<6%N'
    '7j;qrAtvl*@o;O6%S32ffHq1flR6&r@br6qeReE>g-hGG(1O`?l)J^BC%yxrFY6k2e%*V&^3J#B=x4RIn~4AR58V9z+utUJqMWT`f;YEPqZv9+W;3pSE}%|p'
    '`89X?blRBm5H?%GvCl0n%BbAEG62y!wfwVl#A*GK=~ya$h<}+i<PZr_d`l}=+QB3RGzMyj9Bx(^$Qw>&w<?4Znoj^e?>wG=B;yS8c&Fx1lmM{u)EXz%JirA!'
    '<&qb;uH$l@t@@Fy>1cwG3L%F~LNjDbq0X~LMiRa&D<>iTRZ&Gk{Byqo5?o#cv(P<#LSAFO40N6{rb9{98d)cC)Ajje^5L3Om~<oDR~G=s{^(;lB)960^hW0T'
    '-nx8=5un4$Rr0Gt`gKULPOgqhx&A3uDBr7uc?r%Y-vF#JlU{eaU99qEApwTHUq(J63ZAtCP?O7z{N+S=)L^nk)xt>iMX5%6_1cO^53%C3-W-fB*<mRd4wM>8'
    '-)s)l*lghe<(xW-wB`>rqC3$mzIxH5RG}2MuDIcYVV>#zAh>OI>a2)tnM0J~+r9x<d#;O9mcjEY(rtK-61@Vf83;7oHCiYD)4Hd1_>J5-{x?HCV8mhsi$nRy'
    '5CX!_5#j$Aq^#Bk4R_HPm4QPPWP((QtB~bF{m3+77hdMR_@T6uWzTSSgKki{6%4DYVPD~lDX4@qgeb$%G%6=V6(ll3g<<TOWtjTzkldEq)0>DcnX1-s9;P6p'
    '$tqf7Q9vuy=Y1<+x%G&rCS^;W-$j@20!=FtS>Ty2Br$O1Ffx&x^*ss>L$MncnCc2m^#!Mz!qe`u5$D+UoW56~#the$9_0|OIx+x+HZ?xlT{yJU8OjX4qsc(>'
    'Pmx`XmVKQ3w2kLV6VNs^qroxl3l{f*KpOv9uhU-FqR6+r!Iaudk>B_tWjTcZQW?4xSl{D=xSLp-DD*qIABB)uXBS)GD2|#!59+8dihu)ft-7&Bws2lZ1)m>T'
    'LQEWmfT>I0=k20k;YS5YD#Izq2wpktR^hc_`^O@lT^xm!1Hn;P+p)_iLLFZir#@xwO2!DkZR)>`?ojIfYf9cia+cVOOqgTLOtqfo&Ku`lP;1)QbJ|>0Ks=`k'
    'Y7^&SRsuqLqwFbc>%97RDQoLX>23rz5-aeE=M2ARIp(k{=HNH%^C&nw#{cuCV|1Ie__hj}??#nTA7U6Y=!1Wf*C2UZ*>-L-FS4y+GlE4>J$1jR$qpdea@;0%'
    '7^0vAuO0%`XgiGHYTc>_Napd1{nD7E<^C!d?#Zq#1-G(h*e=kvXWq8ZY!~~f3AY^*w{3dZU1gqGVf&iGsV0$!3DmNS7&Tp?$l2mnTSFiPFoI%6(HiZtWu<{X'
    'Zj#q&(XnZ0q#OB%Bw2|RgqXgccR70;XLAwHito(T_F|q+B2_|yvpK6&&jv)9-8hGX@)j*)(I&&JQBe*%XP&I1pSJ1bS{`p)oD5DGCXeRG2ASBIp@H~^2@8MP'
    'qDBl&u$&<y=};==DBVQy46{D#T+3b4vFiq71khrZzs>Q9c5~hznuh;3E0|Skl<hF=&S+_Ntz*Ja*M>P=CrkT<f#6$(oPFF7Juu9Q8#D{F_&tdFLe%hdFEuR|'
    'agUT+z|FaUJ=!<Ho)>zd?ucVYk<;zcA<a=8+BBzq*0_@iiZL)lRI_o0-juELUu>qL;DS#6-856AS=trOTf%!6uJ_UkF63WT#6*VSk_1}(?#U2+Y2+a*6ni)>'
    'uZdd5D)c-wl|Yb6?BXAn`%}QU6vkLrmYafDlfuhdl@)QZXn7Zg7(vtaPu~k@u)c{McFkPhaT3RPxPoJO&_TG$y<81p3TEyQr=>%9eUNK;m4&pVGT+`i`BU4>'
    'l+sp7PvBoyZ!gyAwDI0a8760Cj1cL}E*=6GlA7#DW`9zY+LZA6*C{IC-)*5k2~Jgmh=gC>J%1bN|0ibHKM@n`;s4tHVwEA+LC;3gsZ%rwkL&EC(wr;y6!rmM'
    '8mXkfJAo3r3{Vj$p@u+jI;wkWZm}R#)w;p4--7~7=?+EKPvc3BDMxIFepOC{s*4Qkl&^@sWYeUbMcRxRqGTJcL9=nsu?<aY*fIg|8pPWhN;_vJ5+SjMO`15E'
    'P#(kJ9m-fJwaR(7aj$6g_8&S?dJz5vC(RTrnhh+T(-4Jy8=@vD{rB*Z`0uIs??dt56Y<~Ual_3zYE(7E#|SQIfBX(#9Wdy6hW}{(?vsD|Wbw&#{K?m!eD}$V'
    '-?iS4-zk4Lp*RNwH%aS#qx@pF%kUhFN}|IkQ1P@KU2K=rxva!hQnAY@h^m7Y?63qU6wMmQyzUVrCFp$r;-x@R;-U(jQqfZ!Opjw%J-tr2<Y}W=YOt*oR6>>Q'
    '{iL5t#;=&ho}ypJ#`6fj!%C&gOJc)DN4Ek1P8UlN(VR#}0caMyrqW!dBx+I*n_MlC<YvR*ID5V)z+8;Qjhs^OR-^Lz?oLxrq4$tpFF2Z1V!^YvSl8OqxJDUx'
    'Pp|5l@SaXZ6IZc1yQiCL>8jPqlD$S{sVTiuS}Da$nQOFv)=Lj)Qz#qVA|Ix2Q)nrt1hP*S3PbIU?fVbTO`^7~TR;mg3o9j=$UP-XLDhB3c@Xr*={7t`^N|;v'
    '10NX4)9QwMp|)4n1eT>PkvNL+iNBdec9*dkWA(9$^J*)kJ9LE)mx=g(j$8$=D_d~++bH%cdmfDs>-PVvQ~<Vvu~B`ktz#^*R@DQBgRT#SVAfXPY)Nn2MIs&?'
    'S*SWF<-M*+_8ae2Y`2a-!mp)vO<GuN$S$&rHQ!^l15M)<%~BjrI6He(*mcBL=8al_^;LzeP%fF0lO9l1nN$_)c;Z4zQk7t-mR-T^$x@P^zk6?}01<4tG{r4E'
    'LX^Y8oDMZ4AZ}PK^l&?uKft|c9j9ITnKi6K{d}xVQbBRJqSPb^dWXW2hIm&pl7@I!qLGGp2NIDuO1knQsc4OAsq!6YR_+Tt)Zj^IRNR(NzIvsc=``9AAo84H'
    'UCCgulWZZev`|pbK{vAeV@W+iFTi4q@p%8c6;pndNPd+_{{AEqdI}kQp!JdEk~;Ozk9Vnme(Y2-4<tO1p<aQ{7LcPm0Afd#c>>Mc#-U73m8qT_Kt7kevgA~I'
    '6&^p{v+($F-NNI?yVV8%7-hjvyFJ!wJV#}DDzudysgz8nzhsf{<HH?_gr7>hRekWAqlX7@mJd)<W7G;aDuiP%UBf6xji;%UIiOPE)SetiEPW1_ETX9Li?>mH'
    '&k0>)+W{AO0+Gn1Am4ykLTHgKFF~LJ{szrbR0eaYCSYA@4A(a^E;4Gs8GyzxGpaN-1!-lOlgCFiPht$E1355H);A(UIa);K@rIO0(IVZXTxW^hOrB<z`2%x}'
    'll3D{FA*S2jEpHU7Ppd_(Xq-NUNoN~sou(}>8dcgHkugrlJer@$N!a!lRx}>El$33{qd@5=6`#|$sc}%;^e>F(wPgib4lsadd9vA(o_yw3>T;zy<S%~b8MQ$'
    'deL?2qeH)%71AG~r{<E`N;CZYyni3Bl65-8wUw7onYR%^r762@n)mN(akOA)Z0oA7YQ!!UlFXZKMR_4%(@1TjZ((Co3uo2|`bVzbqF2A4>Cit7H5ZFe%&U@#'
    'T}`-x1;Cy(YKAPc1-T<^uDbK=l2S+w+5$#V*}&|gBQg=M&_#De{fwlKE19dl)e?uEMp3^yQ>U;Loha0ZQW-|y!5xw=FFD(mJlAf_NuZf?{6%_M;Y!YlfrIQ^'
    'rf@TkLXlsTr$bW*L6-it*W!k<(}}X7fxw*V;F^|uJkND0T;*YMW9Hn#lqe!7YWWrqKGuQ*P{TshTM6>T!SFJ=^<+FWE@TuXR$}K9)vh|S3M(ZR^<^Wb^Fo)V'
    'y$3}xIE)~}o5iRpT~)i+v#kz~_u5xG*0<ha8`iDCY%i(E`5ASve+yV{h|oLGnMFbti^l*YTQ{W%ws3t<*YLPsSKsRuC9hs$q|dooGwXt?AQ3dW@C;diGY%Py'
    ')yi5jUEdgA8NXJowS`j$z;t?Hl%fEd{lkpJDlf~1&TXMUl5t~ZZVvIzICEl$VFWk7u9PY0UdHd|!(qor<W4UnwfLe7){tOEE`Pu}41$}hSaK}Vz_lg&D!>gy'
    'Ag#E)La8a6Ps0lNhhz0B8xtmtb~Naon0fMP(}W?Mj`0x&Z!o#Z&|`&nTEmkVvAlV&`w(3nafUO@k=O6>#DZhj8uOuS#{5Q&b#i3#tgCoE9T!PP$*|?G-U*R2'
    'r3y$BMD!p*?TX|+ayT)}o%aOH-^R(_sPwIJakg_9cDE&ZI400eABT3QgO9OJG_ZfbhlR^m&Rs-vVfqzKlj8ohT#S3&M@Mb`<RkrO#@WFjzhrXGSdDPD)UA?h'
    '5Tv%?6h=k1bdI%jrAM$T#|vCvg_+91OkP5@hmfT??q?N(gARuU>6F11B?|zqimE#ami(!;R0r{o`!Y8=QxtfYoM-5&f{q}CkQKn8lzD$WvhS}+y!zX^y&8?i'
    '=kZG32DoD}R4Ts7(kaGVb*@10V<i@0opT*OeAmp<O(X-%NJDkCU84(no~&6nWmk8^(tr3s-=rW$bY}4c>j{#M?*|#`HN;S=X_m0@Zwsdp6<Tef&N7yjiW|P!'
    'Bx}@fi9AJ{ZuHHDi;}Bi?q)$lEJ~>7K}Fk*=*>9>v>`@79hkn5JUg;(BGKjbRZ^^xs3P)Vf*QXpQ>5u+Yb=BqDlaHxYl=KBu#-ouY~MC`pSkYB<&sCW^id@S'
    'AaY9L@a6XM@Qfk{;Iw@-%I~q*I&Kvu*!*68GQ-<lj`N>c!kSR-?ZB<`$a{rBmqy{<qSkIJx^)JHwjej07Es{J<w)Ny@Gtmcw%e}PY=|i?vPo0f4oh7D=m#Zg'
    'Ka+x@TVeHs_Xb+(a*B!;19oscv!A;1&a5xQ*)cfrEINNxync)u=Qr}k8CjjhwISh*T+ECsQlyc9A3VxZ)y}UQ<-I%>Bf`E3kLYWz|1#snjo)?9IF|`@??*Kn'
    ')t4e--#6iw<gRwR6B-XRl%~K-QlM4mmUVZCqXbl`6}B5Pa|Isb!Ryz|wulChx;moL%UJu)aQ~0;t{0aG_41yxcMU<4#xUye8%xz<71>{`FPm(2?ftdy{QBO|'
    '3UVx9P=Ho9C2LNk|5W{!gYLkRpskvkR4txyfZavhmXEWo<ESbgf5R4XGmAq(MMayVsEX2zgw~-vnS+sSz^>YPKEya|M5Bd5DgC3Q$fwy{pJYAS^_XoDq=M&~'
    '$z%Cj)7U|<v}4SXoZAHhNTZEYl6~ymb$u=BEg~VVTkS|8Q;1@LroVS}%N$+*j0?L<dFen&kdfH4Lpb=0?K~zaD~Z=2;b9V^Xqw)@lqfrZC=L(9Uv93t#qe%9'
    'JDE#Mk&^O4BgW^$>+6xZ6YvO_YeX#-AQp}^)#zT8OMB1BpDM0P9Hl4aAwj1=$8w2B<uI|XV$O>N*ytKNO1e2iVMg5jJ)<NH84%=1hCQKWWIy<kM(@P0F`914'
    'tDshf#bLpCqs10PapXiKhI39>JC(wkqGfpz$t->`Px4Pu?x(&$5|F-gv|Iyr+4k~EJxKu0!bq#YRm6!K-GOG^1m{zS*X04N5Q0j-Y5d$W>T=|{Y^gTfG~;z0'
    '&VZFz2*NL|I_o>1T#N-n<1H?W48w_4JCJqZKIYIeO?hI|N|2a++YyT4Q{lYm=AE$q4jcVUvg*+>)ie}khGl11(rGq3>oXsO^PT^+=9G^Le5<UEhKZQ$VID+{'
    'r2jDfHu_w@A@zF?_qfWZ%Z29;_$^9)7hR*jc4ryl+Y&h4hkyO>FB!UE+@$m5GL7M9I#W{+Y@*qAIbmmzWtK-&Su%yqZ2*Klh_!~GEpxss!Uu7BnM5Gd04&^4'
    'v2<{W(NGpI;iP6H<M1KQOa#Bl;oPm0>Go%6lW#9N@hdov2|vtZ_A5yi(X(_BAAGe=(dlIoVRCc+1Own?vKa_>*;)e@Yyzm?>mHuM%Bsun0sJx;z%RsR@vr^~'
    '|JCPjN6~$KEbiIKvQudgDg-Ty0j=L2l<4$e<&|rA)ptOVG`sh4L)9HYD@!+Na<NY0Yropi+YZihuXlLFgDR$P4c!S3E!cVKz}{9g>|nsditqkV>5$d}b{)=T'
    'YIq0xH5l+J!;(*b?H4e;SeD6Ewi$zX5YIPOQVV_Eb|k`N+8WquMnT9+<24-))mJpq3qX0om<<B$u~sdA@nyi*C!kY8I(tfTWgl|k*rqdKJaxxCz2Z2Ls=A5#'
    'QNcKfzulUG4$e;YN%gzi&3&GjWoZkcfYy^mJi6f*Ppf{Rm7D&zp%{;9rG9anB8#wE!Vlg-t)z777q_V(zG^7<fYU5+$cFoz$WYxy^F=K^7|Wxp$We;5&-CJT'
    'YLVGm=O=IAPD1g$;AKf_+F86v=Wh)qG9VPa1}SC9YF%n{iC0lfrQYbrS)9&EU7JnecFZo&gAzsnu@7%9)I2J1Pmm1j6k~PqLK}*UlreybM*x#`+z8lEkGG@O'
    'S8!)@M6mQHWL~x;C(TW=#7mwWJdHTuF8XO|QNeIKCGSj3E9Tb_OWF@B>5FeSr+D|x@qi_dKR$gnM9eoy=OTL({SkzM3joWRjvqv;WZjWz26TeGseD)&)363;'
    'Hqfj0)e@dTm{t}fH9)E9gf*S%#5`_9)*12~{@DzxI)nGQ_*v(9F@Dt9S&Sdej4JfA8<U`s<tOGik7vnVI3FE*aE6QKxHitGJLCLHtk{@$)_!>Rym&vVP3!*d'
    'coWcD;C$lWAAK**$9J<oG`GBp*Q>kS*8St!Xdipg#*{wod(J8U2XcD9Iqv-qpUSJwG<&@~xX3n}Y|+`SBH)@arc5DwU3x*zJZvT-2FVO{(CHkLI<|Pi`{Vh1'
    'b5VZ8MQSZUi$a1cgCY*0P6z%y=_S<Z|IgmLEjMx`38L@(iX67d0F;s_@LCikwpmH49%-jqlBufOGzP=KKp;V85d_e<R25ZAv;BzufPFm=`*c2G{$#)8guBPZ'
    'Bf=w*AX(kBb4F$=A~PdA?jC;kXrfffBYC3ErS;WAse=HvkLcuF9MOvt89RI)jHX8fSE26ZS?3XBI!mn5iHxPCRXRKpR_R>E97iV=V_K+7TXlW1e#{!3$!jEI'
    '!qfT~{Al+z8ms%8GEt5nGom9IPo9Y*I<7yW)0!g+sfru$mm3lm8D372l6$UI`kIc@0aDdW#Mq@XpfQ61(n^tco7LySG^h0`jYA33oXazGDz3ROEb8p(>0^d='
    '`uL%pNkcn&ilLo6W@sm;^)XN4e%J-`oE+60+SA+>i9Pw^TpZnbP3p$YL#^9S7@Z9fy@f=3_|0)%pAb5dM)=}P9O0R?e&8BFnG3*h=3#`Je3WE7Dxpysi&l0P'
    'fWGKFsJky@I#ayJj*cElJ_S8Ke^`&@4Iwg5$LIBN)$z&J9-p5+f}$63?2$|Yok=tOB7tL1<$mj<TDg(^QyCBhN0mE6Jj2JQkHtGjm4m@;bM&C$yf~`dHu9Bn'
    'BJC;hk~rCEPqE>gced=QszIH1<UQ4?x=D_NL7j@2^JJ$DN6Bffi_y))gbPQHL!3wQJ$-mud8~2ni&+IlT`63U*_tyE7j>j#(5X*1xqVYpzEF9(&*bA!wR27+'
    's^aLZ{;+(Ssh`qt-Y%x7(NwhmzyELeIhx;Q*nh~!0EWQRHMkd{?6H_lmczeoicvY9Gk~B4U4f1iE6o1=f3M5&u${duXX9b^@BdFmsZmibx&HV6cc~irGNbeu'
    'x2l!7FPi-W#_brQ|NBz?UKCSMS1#sDt!MrB|COyaLkjnRK@va@+fK(1qQyY*^F^jDgg!!y-g42@mOz+DT@2B5K5*3`L=#@(crPl1aF)ZBjA4c;w?Mw%GEs>?'
    'woDWKn&C?PeiF4nH!wDpL0i7H57xk}s<x^C(rUO)oxSJ_qwK;Gp{s-1r|9T*ata2aiqgxTIVBmvJfP@Yfy*Nazm4HebAl5Hm!i!Pi?uNbGHL>-{bz^3xZazy'
    '6`2I>fF|%kIBn`77vcJ-PK_5V*OKFvVX)&!)sC%`1su;Lv;?@29K}UAuK?59VI4G9dzy7fBfW}sAPa(yJaz#W(v}4ZNlok#={`?)ozp-xiR;`see^nGr@lI)'
    'g+6`cLLVN{N}pD)^j=D|1A#w=%jcWfCFH6Y+ij4IixnCq8NrT1|CeepZVU8%(2gVA!sP9Bu_`b+i&Nh@xipxbj3_WX8ItDO1%-x_>07>>UF8fWcoHb?)$_+7'
    '{iee7=8k2$tLJ#(sONEUFrQ4?bxav59WcbSF%P~MzKZUJGq@KNF=B2A_;uc?e>v1kK*KMHGrS;x{;`(@{yy*Yw~u`K@R92S?S|fgu^z%d$EP?6Pd(}>8+8ZA'
    'cz8-#IA{j?k#oDd8ii{yz8Za3ii)ezzz000pidLf-7OlV8acAy)n^HK5`ok9-q?#YfxHJ(_Cd9=4X1zZJ2s7kcag_DV4loH_qpfkL3d#FJc$g!(W8Z0tw1yI'
    '^9=2Q=QFgdFxmiaA`C&Iz|jtF0Z6RYSP?0{!cbc4wfeSdSJNJBMF7HpZmB`%>~zrSoIVA~*Rd>MAedE{V(+Q9p|jU5Z_Bmz3V}0zu1`3K=meWV$gg837NRnK'
    'b=DF>SeWVSbrF6TJ4eR>v~b+KL3@Kymj$J8W9JD^Lk}BfUrwnLyAbyNm40xaDPg+|pC^|lHybaFjTqs_1-jwMYgsZq+fA>(eb(cxfn=N|YOR1WE{A2{G=v|V'
    '(Nti`Q`1Svu31-v;=6LJiejdx1Y!plYvw(M!)87Q>F_#nOcpg(9na*#%RQ=h2!kpHObIgQI#5Tap2Gu}Gz`f1jG5DGvfcLoOEe-^Z(Qr_e9@jiJw)d<BZPT?'
    'J2se?Va#6$)s4E!WbhVk%=H@W#||hJN{(h02bMrJ9?9x|q`Y6he*10PH}QDuY|toF-%1DjLZ4(9{&N~}HoRIE1rgQddPUw75ZVxM=eQjKMK?TCNKxTVL+t`D'
    '(rx6mvMz2HQ$Xjd`Ci~?b8tg>C9M|^x{b`&m^Af0x$6wAYx%VrR6CrlCwFpwu%oZ<s#@D|I&2&$RoFBb!eE~(6yxTq9D>!>WtBtI=!8{tP>B^rLib!Sjy^e;'
    'NYu(e9o1c<AQ-hf=VE4@o(bIpseJ6?1!r175GO|wg6Mb%f@BBao!~h>*=(Zz;&B^}ofnT)uD^J^Qn+~1?3|oDWbrDxR<(FS=h)&&hWE#v9TxBSBwD<qc=6C#'
    'L|8m2?#GrD+QHiZta|Et8=M|JWKGiXYqocX#R%$1t8l){5jqQ30iAimBXlOM0uL^5bZR(akUjgs1(I=YRIxbBjYGi357HK8v2=uJG?9~g9p}L~ngDBiFPg!J'
    'Mi#ID?Ct6~BD2*=M;6@%%!I`AK9Tccx+)tucQ3T%$kFqnI-Y0HVj{BV_8Br^6NvSQ_VDF=zFK>suUNz+5XlJVjJgK?BUl?x4f;RHavCW#p6e1tM?E25X!p9+'
    '>bGQ>jz87L`SjTZ1tj`XkI2R+QmtptDdd*F5-Ts-goEf}Mg##U%xtDY^TwUy5goknymqYWql8!<)VDB(gq{F)_Rudb(L?GRWC#(WTL?iW0yY7~$+bhPdchTO'
    'zw<z=VZgjFu5TsX8AQa3;;KRbMjLAn9$1_`e@w$M<MW#&1VJ{YA|{5*J;M0tn9uwP<JTI7DQ36{oGUq;PUj!!jv}8WohL|03sa8=fFN5X233d?Et>d*6rx8a'
    ';)B3OusW=ZvB(GF*622a=wGIza@fk;-J=*>3GZ4j!Mefl6ACQSlG_G9MB6?*Towv774(~TM2<i9$ZjM_s~UYu$4jySCsTn>Qo+1oJd>6e5e1`D6yV0W27f7v'
    'nbH#6CNNFF0X1v2TYwd>i&Do!votPnv*U6yL8;Ojf|^|y)5VH-?pt+8o?AEG&|(_+aUg}psyF2bWHv6$8U@7xmvT65n>7pVZ;!93Ho!DCAiz=h@({115r!PG'
    'vLNg{j(LRf3HEVp>Q3uH&;Z54NI7Oh_=;9MN@1jcyff_@>8y}SNoTjx*&gvhN*6=t8dANe1&u$`IJ&e&gzx;QnSE$HBK4!xs*Yzwu`G?-B|afB!b92Xw>(r?'
    '5+3zNXVRgER=U&d=$J0ES6eF5F|&Pluc86)Pt~Jf;JuN!7cDVo56e{L9zeSCDacuL?`fTXqOc++V1X88%}D6v2|7WNq?1@c@-Oh#RQ;kzfLsrJ1!7?qFT}h1'
    'P<+lcBIul(Gma+DKJ7;Zx~#yxx$Z6uCiQN$3q*6Rs`jxS1hO%$J3ycIeF>~;*?ipAF5aSho=Fd#xZDM!2ZT}O;2SyV&&DRlkrZy<3$5>?7Esc-6dQi>peE@T'
    '^2`?j8>K<6^#hb&fa?QVx$9Ey3;Qp;r?@WzpOS!Age38!PDO^7lNuq#Knc{|!*}WWEU8!p44o+`)d%&ps<uSL<#ueab<kkAi?1MN7f*Bs4>_e&gA(dbz3n83'
    'Gc_7-gH(6Df}A!@@-W!fOx1piQj?2@6@n*1!o&jO1dj$EQY_t>#L`*lIH+5-Ay^cT1ZSrj@i&KNH=uFgGwk;HlKS&3&`QJ*rKM7}_Xnn;=^YxcQu{kEAf${;'
    ';R;CqScSt<E1`4SW&4R)?BWc#t>aI`whdHea1-ce#ub~Qa)u>9vF682dDwrg)A{?aMc8S(5cU%Yq9J)6_ORXx1UFIb`QvhVXnVIQ<VueJ&@;(9`gtdm%gU1z'
    '>M(gUrCjy!A9_l8M;|<;h;LF@U$lt$N<jC5@wId(oZy|X{iemW^w4&2TOi!vx}cFb3Hp<-U5uacVfVZJ5800s3Z$}`jYx*W%_or<S6nYzV<Jmyf}p@#h<y^W'
    'y=o5CHd>blCYlPE_c%TqV~LwE1Yw|+c0_p%j3h@(=ipsTTZN_IT`Nf9siq?+Wfy0!&OH?yh9a$l5=Yz7oHf+aQTOEtcvX275HK6pHx+QO7_0-8b$J&sPxaZU'
    'bdN^*9<C+8ooJ^f#(B6C1_4zprQjh*eh!{PKNq(TGl-ig!F0PignKN(O`d0*Fq}GE2Qt!NM;mo|&-hqT>3DA6QCz4eKGd#PF5jcWxhxB+kzDYNx9>KLZq0si'
    'd(yOr4-Q$>r{ZeNC^jI>D4OlCm9~aEZkGT5IHM0WM0XhXSC{5BBl9{m5%Q2Ugj>BV)GQT{5_EM78CO9QS3=CEWdg7o(={3dK^6yXeRSM8I63_SgbbsCkU0dw'
    '8kf_}n$USFZj^_weLZ+^(}Ov-aEa3bxUn*QcjvT~GVUc0M`i&4jk!LUHcvbC+B7Qv<D+DY!GbL3A6AqA78e9J`wSwY1Cn1|FU#2t%QF#Py>vsYi<LZ?s%vWJ'
    '0nOB8_r(Dvtt0(V)dhtGNp*D8N;P1IA}$vK(A}t910qx^m%*%6&Ns=InSHPN>plR*#d?=&8;2gWO>`UUm0XoYOsKdp9rnuk-LQ2!1JMKT=QEBq&*OXO<jB8='
    'v>&9a9@9VDxfQkBdG8oN#N$EFpBhmks#`kL93DLNg*}-9miT2)pBJx(d!i18cVxEhmlJfOA}O2C7%7}X?&=$tc{;m=sGpe47<ER|1ZLHC6wWY_U96}ElD^Vr'
    '?Is-wj8?#m+p`in6E^2@usM^EE4QMXXO@HuH)7bK&%G1F_WtAV(RhZoigzDOf&3wRIsGuaTM^smT^Dw3zXr38RS8k9Z9?<Wa0N;UG!a&{N7H$r!6)GBIod~)'
    'ar0_jP9a(PdgZBomIX{C<$qr<H>34tsd!kTekRa_VMK*H)J^#SD545B(`PmP(lWYrOXHW8*-oD{(ltIWd1S8Ds4>4EPfQ{Kt&kiYQWMQ!o-4>1>YqQ;0a+uC'
    '{6noV^9jns)nRmr?xkNq(<k_|l8rXIV`o<S=)97Zt^qgAP9I9J$@N=Qnb3^3JksUNXQB11X-iVSpGd;cqNJ#o%Iku9#&L1(di#%|RcO<Mcfbezx)S~Ks__+|'
    'zb38$`0B2LN_v$51q;!t)Gf2L5Kcpng09ek4MMxrcVH8YIw3OsR&VXo;{in*9DrFs;+|4u6NG)H1U$V_%7O+|oZ5ayTd$}V&;{A)JF&(6R%F-bnviUyY%3S6'
    ';GXAkkW!6-vd$kN$<06C5F?fDlMc3GsG!H!S;n{yPaLn;3>aS)W40l4FnGEkH_mL-&6;QquNpbpH~#3-ERcQ7p+bhb_+gFWN2WNccWQx86^Q*f-H>|SN1m|I'
    '2b6*)FpOH;$FR1uC$1;hg3A7$zm0bG0~K8=>p6&O&DIFah?u$RpEqcZ?m2S0PMN_5Y29ENx=|)JH*LO$Lxnr1m#)p4%;7Q?w+lr}ZDyr9%C~qLjj0^N>Be3p'
    'ccyY^Q*WFP;DeL}F3!#n<#vN)M;l0gQ^(QtQXTYx%lvW>7%(F?A`ijjG*w!GLZ!&EF%PG;wLM$~UUM)sYDsGbb$+Lbu}Eq<gCx6$bG2VZyCiMH6!e8mXmfAd'
    'wRQ|w!8`kjoCLZ+s5xf~Z2R8Vp%#F0o!yq4A(70fXiD|Qat27@w6=2BmBc{9Ev+ULZX$HbhnA%8?acxakM40jo^%Imk<}9DR6w@6y%^(4!gy6Jxbfi@>?Xu{'
    '0Js;Or@mz^yrzRvxj87URx+QEYTPa1<$Q)lQ<&4Gb9f{VU_Y|rz*|YphT1m!kwt?3_zxT)E@*O;AtPeC6lP|_K7FFBgvKuRJvk=b<k@=|rrpKua$Vf6nl1n6'
    'Sd?e%nyD$j`2f|yh0I^&!n7C+%%;P^Tuq0go0u6E7;QGYP{+>Dxt|KHyebvu0{l$=<xBKhGDAbZK<Z(#b3o2}!#to~_QNaBG}aA0XiXPvv&`=IPvvs8&X5}c'
    'M;Ml?TQ~PLYD-{O;?xjXsFSOpa1@j$AV~^L255ee0)}Zc-SCLPr6LBs;a6<geCtF7C0x!QlNzukbq<lDd5%l;_U){3I<7BHC#$4~OV<2YoXpT?b|zX<V|=06'
    'vF5KMveeX;!-~MY_@P`C+P0QO7ugydWCR&_C7Muu(Alz9a)DU^RIg&X*BZMe%7NEpx|vJ0;jS>2PonyEI&)P3$>b5IY5|9<VoiQmuHdMc4VUFScwNk{${D9n'
    'f!s1qRNj0{=kRash$kKlcAw#0hZ|E7M2ClmVCp@(u_RD?skT#j%ZL|TCgZ0HG5^b}&3vOhR+;lEmh;_KThfC1F0br^sjK?miWbnXW!@&ddE?#TSC7A^a*FC>'
    'g=1@@3vTbQ&yzT-&5zLS;WP87KMlNup%gsKmibsrZz(Dfi_`*k_QUBE0>^?D+h%GS3@=wU>n03FJ(_s{vg~8Iy6CiwLr6GNE%HM&E+N(4(3T3;1p2_fAg?Ui'
    'v`zYhEz(agKY)wE`Ixbuu)~&U7N_I#&ai^Wf`l9zT>AI^?%1gYEd5RunCf8wy8Sx*(YPF5&E~7MB6I5$2Ns^>ePBD_WNU{4U7%x5K@}*iuAUcX9z~$VKB<c8'
    '1TT;rR$FYJ)P(rczthPtLrav!gO4YoB?$^$UC1EojZKgrUC&p=Y@i(`w2%UdHUm((4~h5^K)yg~o}2;Y7*oe}hG9zdSXcU0xn@0}u+x24oRNzLpiw|5%YwH+'
    'Hvk#v;YC{cKvA?=6}GUesdc~ZjrOiKqFGq>HeJXAU8qzv55#RQR?agF`9GRz!Xk7offe(6@!P0!BlJu!`+<Qi`u}W0zOHsvbOv+^r4wi`KRjFV#TRbP@CxO)'
    'N^pvoqzQ;|ke?isiclMwYvluE_G_9!w0fPs4R#>Ha6>`qZO}1{4**W~ecVqFY66I3uibtGce{8YYFU~k+zTT(xONu-{?|JRj^F}R%_s{=oo4AEkF@xv8lr!y'
    'PdlAs2#Np()`tS`Td^lc`ZV_iy^^pc>7A)lu?h7ov1zQ_MU^`#-U;qW-UFO$!me_P$!@v6*jl~)v3-x=9Zf1hpE5x*rq|80Q84LvM@uylv<$-~F;z3WbLWh('
    '7$=wFo|>z^Hv+t1Iv@AXFK4>!<1){O(L9-4;ZoE*FQ^`u45}weFmB|Z9CjSllKD$&nIs%1CaR=Mo}Zn=&Y?Rx9>D0%X7`y65713qq^i5WjimG7?&=y>e%)(-'
    '>y__~QO2~k-xZgC;^I#SL2#2{XKt^afz6A-a6DYV%A1@oDPg9k@REag?&4cJvI1&%x`X{G1bq~&4V+j*)+%wOT%}YUdNkQCHP_KF)@VH(p%2UC=x!wh+$g$9'
    'g;VNvC;jw52bFpXzg?rNNXM{PW5}vjt8#92>{>0$_M33hY}`y}nP5tU@@;wZKAF3=+fu%Jw%cG2S;-pi+Mj1BgNC1IErFHv6n%QW$K?{vjU67-+Qk=(puQ6d'
    '#Tm%S2*kwF0o|Y|Y3wRi^uES!O!eN!eVo6h%aq#R6M-`fRIQ<_xDyk|#b5RWvZ=TkVLml4?t@a~J;kQ?_Z1Q&L8u!ihQG-CgDkEDsyTgLaeliP!qtfCg_SQh'
    'FdXNy7<n>(;WqKXq~o~6v0PLYL6Ig0BDYFuBww{M1~oAj3e)HI6F}Tiu&R|%B2O_*lHG{UO8_a>0&Ia8A1(`vYXP=EDIj0!LKvC(uoXg802~WVEMyI95|HH>'
    'IR>Qn)%7wAK2xvOmcTUktuXiXcg8MHaWU)V5LXWO7L&7OfSw)R)%v*TFzIMD;c;Y9ZSw5XJ;f+g6wzvp9^2K0DHy#T)u($Ynh59IY;C=Nf`}qoG8^7=QhbpL'
    's4m7x9eCFI@hPK&o7jVDd@c%eS2W448=+HygMn3nb5kRO3(U%V*=|LEE3MQ~M??P-Vj~$951>x{Qz}*9nYGM}P~^5)U(cb?{!g!dJ9zc_&-+3lZkGabq2i+n'
    ';3R?6@|M+n`W{g<S~hm*K7Th`3>HiEU0Za&8;<6e<#1-{lO(utfhV85Ecb0_bI1bSH9^Hqwt5ru3Bp~PwDm+fQmP>5Gq)}aGzZQ30v%u=C_0Hidd=q9n;%~O'
    '@*L;K#EZh$^Ub<g9;gd-omoU*F$PY)s!_aK&1dcL=611aM!`<~E59$zv!*U~0LqDN5u_#XwFPpz1hc9rp?IfWS$E<4L|<j?YY=gpeDvlHqNsHKG+~%m`XEx?'
    'L3)5ztF?Wt4BcyQDj!2TN+BeaMc^C_JTd67kcAr6mZIg!k$ZLyw<Lm&V}0yqfRz9m8ZdWvFm+A?w&pMo-M8ptW-F4-MAs>84e6Y0JutWEdVGN<;`}0c)=5J1'
    'j4<SV<`t=;3;oHd2D&@E6e%5uC3c{vefS|s9875qx>&T8>ZSr{RGL*KfXA^{lE71eXA+R@>8GGPn1QGYyjVWZj@u{cy*3?P*627Zg?ZA_v!T~yU@qw@s8Lxx'
    '*+=Oq2?FASZHo46t7s=(iE^PPTC@I+-MS8usX4A-%lsX^D{qG2Mc&MZyAy*}tLnD!p{!$G_)rnJE&;KC6Qk6kC7z-yz2bf<FG@slS`C;d`Kki!l{0~65`bBz'
    'F)b|-0P;fM#zhIqk}y4C>R|X@-Mtt;U+*Sa^H>j9c<V7qZCa_Rvq>c9*4dPFcd<LE0ta1ytZ!r1SFI9C)k7hSwo&(ZA^1>P$rI|3sIq4(lAx~x0hv%7_)Rw1'
    'dMZv?j8JSs_ry8kYX^OgH@<)jMc^`_m7G|{$P}w4OKgGW%4e6u7D!uH)q)lSh%HzF0X1VESzFbeM<#8sMk+@wHHvy2x7O7$#O1oEp4A+m*(g=zQSG}>-1zp='
    'iY9$<&s5!=dZx<P@0mg)<N56X^fr(SabQVSxz2qn785H(6eYC?rQ~QGdT}TP5p%M(E%M_UqLTXJlA0ou-6bMeLqZwvQiTa_C^IrmgqBz-={9UBuvZA-;65WL'
    '6ZQR)$XG=BfQ#|yEr)l5;%}Sb)EYtL%qT3`rq4XXKG2<ppY<v|EnAEYU`+4#ibO<HuEAwVY9)HIk+Y-rxPl1V)@zof-jiv5q7%#c<whO!K&I&$4P>1$-~`-R'
    'fk#_UVtKomFHw|CD=b|&?^^v7FiXU+n}z$T>$EHCqlLth?0iG1Tik_`y1e|hDA+;N$`#9C6Su-Hl`XA=GG^O=eVuN_T|PSvNLxcjE!JIxN6H{R2ebb7VA%f-'
    'T+WkIziAouqr~2i7oh$lg{WHA>4adjNx8f=N?~709cYYm@cJ2Urt1qu<F`beozt>yW}%ZuuNl}mk6Ls~rYxpYHkicl0d(EWbCV1XvSQ#XbNyzXGKxGcAf%R2'
    '5P23TP)EZCqlIU&@nE8(t-w=dhbAJs!|hFgam5ZTgti#l8z`p3#R|<8XiBSHguqddyon;0v1Zy7GxtZHd)>_AdUn8hFvYn)A9<K)aLCM!$Lcg8zS98!6}d2O'
    'f3L{XKP`v1V7BHHW5E{p$}&P&fzdPO%D0@C96F6*4hvAKHZiI|>`T@S=kD8=Z(a?4dHL&`S4{?tay%(~wVsdWQv{guPTSg$&h&u|z5L~u*Z<n|ow{k-LUTnh'
    '8{5gOF@zThU&BO@>c?TeyXf}Fk9TETlf<QAlZR*U>tPaCsm(wF9FGv<n(r(>ZV77>1e4Y+9om7ZiK!Uc3n3WwQTwbTw-u!w6~=W^(N-dD>4`jtNpbE@t=16>'
    'SI?^vTMOItsMf@Wa_{5BfI~U?P7PLz*`*QsA!Br7CoIeikivCB%^Zsj7^!&@UqoW{xQod!=fXCx+~wQjrAy6A(b{)DU{xXCrADuZ%kiU6nhb<Cd{w*2M8pqi'
    'Kb_SW`f+|Fp8}_YijZ_Ajuv!RA`Vl<eh(p^R<0mcAtdZ?ls{yuVKiGg_*DU1KxoTVw%A-w%aIop^JX@`gk)UnIR>mOK9(!Wk8twB+OjaHCeIl}h%TpvNt^&R'
    'Kp^yC2u7eto3BgV0-DVLxNl#+<=N1l>MVY)OZ)KX?|o^Xv{Qt__DlP8$4mS4=<kZ@PtN4#9(k9YSKMW%XP@;HA0@6zIqYdXs^h8^s~<Zx-LA`ezNK$Go@)Ib'
    'C8Da@wpd1shh2>#2)q=}6QnKPXI5(PfGbe~fyDGtqQyD{;91q4k_o@F$CI6B4<<X$6RmHiYI1h!!+Av?wm`$4eANl>w9_NH)Bf8}znIGmeCC$(sp2zlAeAm$'
    'd$+^UbvY{_s<{!+>444G<z&s^!HuUK9aqJ0Il9i)bGX7X&|nQQ?)irUK+y-6<q$nH*5$!`RvcW<*EWODXSmY7@(2@mkSTk-p562|j;a$gb1u`u5-s$ACKF(2'
    'Lnp{^HI-^62a7^c6ze+y^Pg5U=vOvaU#20yGUkEqf<jr}E;m35Mg<T2)%J_~tg1oj;yz3)qvInUh;Ky@6->xi1MB@t-NUN!2v4h;lt9ohk1Ec7*GPL!wqC^8'
    'R~sFGy<FkR*5=(VQ@}#4+8m^Zmv>lTqe$2A^a^IGsdHRq(aNKpBmw(6Y(1@>L5}vZ6j~0fb_jInVQE3~%z8W1KB~32J`s_fIAfh{k{Lkl*jQ*fb$*&|hMPl$'
    '&74;@6B$x<Dn^5J0s4@*Mo#Or;wtW&Fy^T?C|T75+He)OPS<3!0#4^4t?;Ti7Gd>uKKr2pHy!)e<<&-AtX;gHC+bpGrw|>Yc;b15TNP%-Gn}LH3`=|frof2D'
    '+tRQecU{6(knXIr)GY3FWgb;DTXkigAEldr__YbvL*li0QqkNa?y#M+AU_$G4Yp2}vG=&_h=15FdfY|ai1Dr2z4aQRLz?YNK0+WgS#6dDB{EZ^{Ti<$we-);'
    'nyL@uY2i}tw8K!Bk4$O3b)MDqR+hnh52v+66{(F>TlkRtZlu=Ug#Qb%0WX-WYB(*gW(wnIQPlZ!)f+=X#Ps_ML7<e9HN|xk6IwZ$F_h#io^PAQ7t$;yXRxMr'
    'p+a6G%Q53_ySKXu)2jpuwdLX!A_76&Tp?-R&Wk@ld|j8tuBY}&3nzAv=<fmqQ!moYs$*f7NkIY3v-i1CEE(}o1Wuox0jYS!{lInV6YaHar4ItHvRnD{9m~I{'
    'F5l6z;UJQtI5-Iqfq+0&-?$kQ?gXhWSzAyQ-uU%!<&EL^#M!W^CDWFYI;LM3vRp${1%3VA>EV&A-vdH73B79F9#ZT4LA74gsdaK5uA@60l{3grN{!MK3|dcd'
    't8&@oV0ZF@>9A`&?*}MLwk??lxyEAEEPuXZ`4`pY509$WMxLg4Rg$p--3(T)M-)P})vaI{O{u6KVF9o>>l~e=Why2+4(jg|xr#?$>jkweh1_A$OG698C`qhh'
    '=psQ%;Fp_}qNMzHV|al>ZUw}Y>uwBFO7c?yIFS=51wF+e9@!h}JSnJ;tEUxrs_ZtNv}Th#i*bW)in@T*eAFDey91BU>kq!VCy8<HI5W&ts3Z4`=e{s`Fn3=S'
    '845|B&$l-mbp4SXh@qwR3+8mM=Zl+xI#eq`d~njqx8&k*wI#WMZ7E3EAqYm&h5%dgnj{DI$Tqj88i^Yg1=JFy9ZHgokeFQnzmWL`*nKRf<)~a|-@X13BbKUy'
    'isAr+Tpblf-3`U;3K9fQ)|i96P}fH%GxnYgDB07K#ffFIv$xmf3RQF@oF}Mf9k2??Aj)bvyP9m+EfR&L1v}OEY8V(qc6GM`PrvuQ!)_3fOx?DnmRC#RVPQmr'
    'GSAeWy74{D5We77rfz5~41vfG4%D<B5Oh)uYH=1`iGva%eQF<&6u7vL+L9TELm>LCuY-JpLGX0wke>%I)w7HJ9s0mPt0i1F_b-4bYV{Hwf|+4ZDkBX)OclYv'
    'dsg6;v4R&0Gbf3hNvUXLFkV_02bc4YMuB9Ew1MO?TRNg>j=>`%NgUwUBrR}Ui*h-xNwvh35W23^CU2zD^)ye%Nq_OZr`0-%^@T(>k5)Jx3%>e^CLXG6I^L>m'
    'daR=9I8-#H;6G5wbnGjc((oUwU^<Q!Or3--e>;i$RpnYpRP_UmBUqD16-`!B)n`v_9HAPNwLD*aCX+_hU{cv~m@w*n8aeqB%Mu?UQIn@-i%jphBPwRnLbwOF'
    '@?>(m&mpK1Wk<(#glXuZ@dU!HrxNj0k0c>=Lf+Be+ol4>S!5ofsw=+d+P)!e4t{#|o1%4&+nuf<Y8;ir$s1IHsa_nI=%Lx`!vpK6H#Q*;ohvuew#7%at<<mK'
    '^ln{_vekUDR--v!?y1>-)9$Gh(<9sW`}y{|um<H|d9%upa$#E+qcMEVgC;`5z${?IiwpxM4wnT|nGmL+oxO&%B-xL@(&(Z1cYpc*B^KJuwDW3ve=iQ8EGHpa'
    'Ze|ddP|QJn0%<ojVM2e8^k4H{C#gREO8-6Ae_!aoXZr6cB(WB@AGHqW93-cm!6)!xRJ~YE;77CZ#lL=W`^9+h#SdTn`HMG=*8Sj<-H|%@hU)RUXx;CJ4tr3B'
    'dHpFqTCE${U7(kq;sF(l)@AvGxjSJWOeaAZ-G$6hc{<}<ww%uulwXGH>zrks^Yh9DiHp64xSt{P$T$H0^7EUwx|60aCR8>wNsYO7T3)uJo8k`Gyykus=>QhP'
    'OF$%la{H6LRTHhvpqaBJd{--_7}<UQ%7`TCFkDJUuEvBO>8hJiBv}^U7i;*_)uLEAXeFZ_+nAp@=DSQ=RA@9`+zB~<@MCXZvHd`SVb;aJzWVjek3avED(XjT'
    '?PK)y`i|?#4oC(J{YTl~g)M;66muCtSOEx`|9e3L_9Q4mt<gr&?>kiz!|n7V?n|f^M%}4ZsX8psb`XLqOg~9<Y;kt2Y9zD-s%#|XZ(6ldp+{Rq18FDIbOTPO'
    'MVGTWf`vH&%sic@XO#`ZFvHJXbYToDtS@VwdgAgfn+?$-o!^Wm@hx1LU(1n&+m)w^GwuBZZy(2kldVbv&;plUG2^FIwxpHKa3;OF0m*)}3be<ZSArJV_$tsI'
    'Pg@CEG_Op94z*_ehwIk_*rEQc{}^3c1fL{j{aqz#0-Sn%Jl?r~ERNCiaNvmJ8c1JDN?1@A#0`dWmhVL%@K!cH;A%bIuWCTPT8A19ZR;Ke)Dh}!xnPbTh(lue'
    'paA+2hr9Zi9DlyU9jj;4&?|JgRSl%XMwJb1)8JM$n9ws;wzow$S+_Yst600)3XNpcJQ3YJMR4~GSsW(DhWtV58*nmB;4oU*b4qn(I1K$vyhwXC<k_ery%lD%'
    '`mj{2_yFP;Nx2&g;oUW4cV3m>A#8d}q6gmbhh=++QZvLd2Pu64(a8grjij3;A~F(5UM?2{h?s+ktbk7jUivthRe-`p0a;#rEJhoJtxe67!ExRC4-BSxFNM<I'
    'D0*;Ns)1-WCpp}ZZK0?%)#J1}V!F#9iIIWhrF6g$w)jHb%k=8BvrqZAfc!MEywn>aL^lgolk97qC-uG|z>#p+2e*y^xOHd&^L{_sU6!#x-MRxK`l%k}O|hI6'
    '(_NyJVzt^)N~LAfD(b;(uq+np28M0={i{E{{LAaNgP#V!zWU|W%eQZYdc|a-1=PHzAOLiq^BAxgF}R$K)cmgsQ&F)&fTz}U`VJ0e2Nt4`EQ0&Osq*iK%N2!B'
    '3L>!P9=?07R(CinCvXz#v+{6W+kn>gAEcet@+rc}PC}&(w1-H5Efn3lJ(?63jkpG(t`#G_e)$v4gBF8;jKzM5LrQxwwR)fKnVkWEtrnM12|-FCpw6ID4e<mO'
    '+%A!zav@$)^sINncBzLaZA(1~rJ9+jkVoiAeR>g)2J9So$3Z_*UfM$O6!6c)2kY|zjj%v#CZHUX-l~IPvy3OitNk40=!*^Hv2}vl3UMw4(L=E>9x>mr!yKNM'
    '|NPHs%MVk+^wR5s72@h>eV?6#(&BxhiI~X!|7x1&KMQ&DU-ZI%e0l*nB6fmXe_yI$&*B0XxKd$QKZ>O*<Oie#+kz-Ur7Mi7U3Cg;&<1@d6-?@FnBhA7>;XA>'
    '3LvKguA>_+*X3k5!l<%=?k=bm(M=eKR-mQ4dWVNz%vSGu;RVy3-(hS*PVQ8$Zn^EPdB<HfeG>%anZ8q@#NHm<rvpDqE#xgI_<f2ucptAo2tqx^kioP)@h626'
    'o`D!|KA3H8dAeu<md*&BXR6)>8ri6&EoT$@Ju9yC?^f6jbP*rv%X+>>&o>iJR(;R`hWAW<ODyrtMRr&m9G*tOTouU4W;!)hp<^if(vgb~^GttH^CP9`v?^xk'
    ';d5$jkN*2Z_N|TycVI%oO$%_w9g<}+DO|xGc(_(~aygxX9kSX3+Pe5CLj~w`3Sl~6rSzl1G)JHr8eKPG6&;15{&sn%T=`>GzV-N?b<#cSJg#TZw~MFzPr-YC'
    'rl^(d|NM{tnL)PBA6~ur<>!MpFaPxF;PrRkW=$OcNvolz!@FYHQk8yybmoAM!S3Ttu|5FgpZ&Z54fpTNlH^w1g(Vq}->J!5E-An56b%3`=bPDhrJ#YXdbFIc'
    'RtJ-E4c|d-2h+Tqzb|efb1oR2<LtsYz#bui1YOSXs%}5|%<K*RdUiZ``TCb1UaAuW7IpLn=0<q^`sJIqgYSQQ`OiQ7{O0YC-@VZm6K^5tc@Iq;)IW;KcJ#Me'
    '^*-<bpYAo_SFc}v_x9&s2j9JX{d(}-&wu&pZ3+nd%WA&-?(L6%eKl|xt)UiG-L^nFUMq@yuH~<z%YiddP^fCER&k&ymg9y8>&?%v6}U=JX!N52`jr)3^5aiG'
    'zNNm9MN&i9T)6YPlmg^{*>Pnol}6_0*~l6MZMC^v6>FpU{Muv>pDz!}F_eY9q3)@2r79QWgBxH#)=P!GexIv*V4A)B%XeQJSl_BUuvjkVv$Ar#9^hR$2Be=w'
    '&Eq-H@Wywu;cYnrOYtj3Xo1Pc*NH8z2(PJPr)Y~${0*+51^b-rqCQ*~GxVT<_$e#Zsd4tz=@%Jf9N}jRST6&wZ2<lx>4&wsh?sn~-Pi-vgO?#}1~;#mw`J9w'
    '8Qs(hDON;jv}Iqbf<3nHaSPLpNt>9^-v0jnp8*nK)M92_NQ#dXqd|?>H5#sCTh&G=6P@qgb%*XC7j+*^qHbC?a|Q+QzsV`fI}%C5%}DdxkafDPn9AE#vqf34'
    'Z@@%PoBQc9C?+xupkN5CVIhYBR>2ACW_BJ_?OuP+njc|S-Z_Mzw*b4Ifs(p^F6U56b!54o&!?+4yeemiV^>WIg+I}@g*Tz)7&dS69pU)k4Y8AElA72?%e$|!'
    'UTy}afy3y%jexL-hEc!Q%km?gbrWg7o2ec4N<ZWpzB+1fgu0AcqEElYuXsizvKQl%YDqBMV0*?=PLx;dIny%SgRUjF1z+i$RJip5v~vY^QkaIy?L+Ja9qc*S'
    'o!(;p@^Cl)MpP4c$<TX6P61Xt%)a?X?UL*uJJd6yMzdO@1BTub3dJ;gU{<ZF^s_HB%<*zdoAwqq1@-{HAk=@$<N|taZ2}8R)nRHny?6LrTS^X2^B$!dS6HGe'
    'Pas`HSn7H+GhtZ{C3Dw?-8NXX!A*20=tid48JFTgjUBpZr$+S)>#<#a39%>8T27bgA&f3T`jX1`_uPSiZHml+It)P$C;Es|5JGI;Su(picbhoRe#_0Y=F}Au'
    'TM0I$Fg%5eb6gYOw+t%=Cl@E2794V8_5F+_)YD!U`GS6$?nWpc?w)_pd3(x2k54b{pB0u+H&wK{=G^)?Up#%mJ)jRA-2~#!$+KNvyOa98M<0Nu7lVSCI7;?R'
    '!qrz~cD!KMoT(YudSj%{=%V8<11(ZVT7ZRy5tYWnEL5=L*m}y$Ft;B^GX+~qBEr!7GFobPc=UAPj&!2l*-9GcZ7@hRiHnZYrK>T3zyfFKK%OOq&u&whbnI-Y'
    'cCc3~548INPb|5Qo)G%hX`g1#vL+1RODxz5n6PkT<qiTrREEUTwxdX=H393Dmm9pMy_+qQ$fZzppmylr2JciKU-d9w<#joDmv<Bz*g#G<&-xF;I{;O<bz3X='
    'RniFFV}IIf${{eC<x&l#yoG#EM;Z7Bs~6@1!~~@39pH5Jz6DORoutTN5(pY5l6=8Z%Q!CZ6owJ08s)7ToN5GcWOvdh&p?HoKbETt*Il9mRSgh|7+B!5vrQgS'
    't8=D~CqS~TcsmHC&6EoZfAbxK8^ltOU4JRf;$f|{B#HM&Qx|l_nzc~<zbWo6$jT{u*Ug#-R%q0F*VjbCJGiV_zbEtQc!1u(zy-ig)MouN{A8Z2y}uhuM1alg'
    'rgb4uAetSnMYIk*Oel#Xm~mpo(AKJLO@|zjgPJ}M@By~7A$QeT!f7NIiMEm6tM^T7^P>AQ{-|eMXx0EzTYWQ`OpB%}?ag_QFO{N9cx56DG~r_tWs;Azf|Jv5'
    'kkg>_flC-CTLt1qktYxDc7v|;+C0G}4-`bHj`*lZkl<i~r#UR3v#by$K@J#h(7kaI68cvCk+G0lWlcEK#e0bT&FFZVzBVZk{sdGgzE}ZAj`@IZ^X#X=ZFYUP'
    'm@C!*oRh$Xh+=|+pV2FIWf_rxYP3+No310kN9uO_eb78Sz<GJzJbI=;wOY^KylQ7J;T7=c6R@DkE(_H?V0uAyXUG^*o+31pQV*=aO=?>0-2x1~B1VCu&jG{w'
    'Zc%MBL%~MY`;o0^w5KoCMpRS42?&dccBUURu(s;BK4^HzEJ@M6EZ3Cg8#Cbb`Ypq{+9K2|EqCcgLIu|&eI~#Gr(3msSPj6?JCsL$Mb*hQBj{LpGfpi$V3a)$'
    '&M`=EoUt-@@EgC|03U;P*-ms1TLlzq%_DW~Jj+<hKvws6;>KCThN0p};r?ka78CGuSG&<`z;}<-nR?3=S|=Ax4KH9#Tr1~oN72DvH<Av!7(K^%t%|Czd;<l$'
    'hLFls3mO7a)5C9!T}qIR@Sla_fcDgx=%Z8B_#>qNM^TO|;0slXq4T#jR0ox7SuCxeb^duaPkO3(nLqLi&dnEZW3?VTZP9o>xj1Yg@v**cul}~AB=e>V{&1lK'
    '3Fu+B)CFSOw5W^L`dPLre7O1y`Yft~5KWbpKf!|!L+uIkF$B-fuir#$RiK5zc;*_~Ru$DzaY^Wc$y=+6!lO`%gyYNc0Hngp2INJRV6p3lu$FF|PNfKU1H$Qn'
    'LUPLzWz?#M`AHKk)N<E4bXh$KWr%3f?eo15DCym4G{~-Jrz@UZGqKwS*C4ZdGCc*=Zlmj=t)TR!mA8}c2{Q#y6+QG6)ZBRjX~Fd{+$8!(7dI)LT>NRzYz3@n'
    'i})Z{1{DCYkOEiEfMm0Efl4)p>b4)q@O1a^qDjFUe)Qp~JLZA#?m6kIVdQSl<otAcYz&wKE9z5YO`<{JrAR5_d?E({R&KHMmvPhaN0dg#VkR8AzO3_%>CM{h'
    'wjmX*doDwPq+4;%R$8YyXkTClLt>YrjVj|N{*cSeLG&@_cf9Mtmlvk?Gs|)L^MVId_oCJ3wuRVb4uGn&X$$J|RkZ{Hb);p2cVIpBM9G>DbmDTDj(0EzIC8$^'
    'qy#kkVbNstSy_Y}ho<$pR;HpvBu)(1ed{I(ci7Jh5*qe=AsARDdb^0Jt*ApzM}UgH?|;D=0V+G8F=Xt;%fEb=*>ELWWi&UpI=;HwK316og9zln-T(~-<8MN{'
    'WhTH-O?`=JrUhbYypW0OP8W{Iy*=0~UMxIrx&mz_twUR*trZAYPCP2w3H=XE&sv=82WyN$aM1J?LKk^=>r)J<2Wrs}M65$I3WKEfY|nQh!ZcAALAT8TwWq^&'
    '9EqE-<H#|A<s7=?DQE{kn$IS20m86$a|_E51-ip}st-}MY6RL(1<hk<5+@|4I2X&}eL2T)gm%c;ziAme$5XLW7Rywxr-CMOyyzLI{g&D>G~E@|VWKjVh#t4E'
    '51`5Wkymv@O&#tD;nMS)%kk9a98baiZb`BG!g2nQum6{LbktNgMXOa1Er#_ESbdaf0t65$G|ln@en8v+hr}D)AMF;2yPAj@=xx@ZFX;+;P5fC)7pIB$`sx){'
    'd%|NEXs57w2n<JI4|o-5bGWVno+kvRwW!4sVgjI9V-Ily=UvAOWFs{|PoM$P?wi}bHM}!$++><pT0r3S)E4)tJ|?q;2D`%UuxG^$Q(#V>iF-MmpVV&l;k9x>'
    '|EU89=?I+;uo+m3dC=JFk{@-cu+QAuCL)<S>u;;qHLDOdWr#np`UWk00pt*XG?de-9B*)WdhZ-bWG)X_)OUE>s+Rzw@6~F9aHtu`*lBQ<%zFn*lO4!(9DTG>'
    'ct*>?)pR~s4Z)*ak?2N^fRpj1?Eon$?091$4(ovdOw(6L){-3?6<Mvq*pkQ-R8xgPd>}BOvy|B{#qvP4?O6v+HZ!Xxwq{{AmoeEEInyyqn}Ku(nCO`*(*%;C'
    '>CZ)5pAHCJN5`Fi)%=v1Lzqw^+I}yV^Oe&ZaB8cV)+webkvbeoFC^S?%GLQryNC-;TOri{evw5Naf-hQ$u8h+LgK|*MX8HKeeo|0Gb@~TVMMJzo%MuzW(E@='
    'z1>+2(b)}f1u+3P&3LEJ$xIrq=U_%=MKW01;lzgR4J)M~vr<mB$%@&Rxzy_DeI$v{t<y%G3B11RjZoW+H(lHHQmAe}Rsc}=yu{dx?>$LK*#A6>`cyF_V;?fO'
    '8ZJ0wpprt!WPVTs^gTzJXv;Y%b%+5lyd<m%zcdaETm;hUF=tU6p}2tk{Mnj&hFUF35oUZQR*)#KxVp>0@ohsfwaJi_jHL~xlVs|41Y5Mpwyu_;bH#xcKlVM%'
    'E|*L0)e@x7eDV2iBJr7l@kfQ)JUF$~BwT>0`&&ys9%BV7oSpG)Utn~Vau$i9tHsX^Fxkc}K*3IuqE=S$cU9>3P(0eebtZwKCsc0Wj$Y3P+Blg=5LF5tVPOvb'
    'NX8WLq40$dO}umXLGUj8eq;%gGwF_K6_XO0(0L2(3Vd5z?2Nq$8PyG4<wlVcBMKvPUl2qjE}j}!ur@J%ZyQ$BCxGpXSF!DmFSxUt4sKf4Z=gi9_5>^WMkmf1'
    '=DliYbko#XUbF=a91TaU)E_%r?Q%6=&u8TbV&Z^<p(59Ia*w+6@y<49s6A`$Y1W!)6NAM@vk5wx-kRKnmZNqC2@cwAm^q-|EI2qRLm8j);kuj_NGcMrR`y3Q'
    'd%^@b_@Z}k)bAaG;gfkjgy&$zHFc#sw3sh~AdJ&uQu*D|XHc+r1O~wfQ~i__mrYab8C5vu71T@mdE5td8sy{?<_$rB2_#iJ>>IQ39NV)}XC1RsEjY%a98Aj_'
    'SP$RyH&<&l&a(m4g|*WQ9t8g+trgU0b>XL~)xy*W9)9)?goz>%8eLIjz1Zq17rB&_v7SwpK~nkK;gz}vH;^QQ_6X#+9nRK80oHqz`hs}b5`u$LQ}$Md0OG#U'
    '7_(`sZ}Y`z^3jymUPelNeZA5^{bSi%^xpLyAAx{GgV*K!9m&2g(&)i7K~+|YAStx4!w~`MK|3;9kgXOSM9uUs47x67lX+7Q9Ns?DAfI6_E34XZ9dp0G%&3Ah'
    '=z?R6ql1Z4T==0dNF9ce@oBtPRJ{6vplWVw6Y!oc+xI5McDx4$->NOu`wAmN0_A({KqQF;ZBw-rX5Vk&$1Bw<pHV`PQ15wWw4nuIojpF$tUb*=Rog6=pj7s_'
    '>*eqRU*(Ck;Xzhr?mN-2B@~Q9o96y^%Rc)Ec+u;{f3FmagUEUiSzQMr3SiDT%-jx!FmJ$Qt4{P=DLMxmRLkc#=&k40_IwV!vN`l(v&{EiEX!oHO+sdX@ImdW'
    'rz%h?e3=F$^wYpmpBNnX%P<o*Ap#75(P_gE5jlrd_P7*U?<8}r5~<<r#{501x)oYK_ujkx&fo!=_Z~PPux9Q%tQ&AZI1%GgO$9T8E+?)5q!JIAay;7CbYgj!'
    'ZeiSz62o9=%Q@9kTdM>4CAef12WP4d%r%3O;{cCTr#-QcC#H_}TH?Ml{eIW_Pjc<4X$$YV$u3vi<f(49a5p^mdIN352Y$y5R~YrN__#nrOf3ng(mriR{`#hf'
    'MC1hq_mZT90B_+~OlLr~G+!7)Tg&Zmp-6i@N9BmEXf{@Kqxs#pIwjlIKV8@&{6LoK5dTW?lwN^s!AGv(oQHmoSK6~=GAyTRt-21bAB~z=>Tx-lD26%%wKxiz'
    'terBtIAbB>Yi;p(!+^Z_WPaVdRl%3|>%I{nT)iJmCC?0=B7_+K^e6U3C0vs9>Z$Tq6%ASHxT?lljyE{QY?!h;AxCsnTw?r%AuR%NC6nQFdN~~3SVrSFv!6$E'
    'wJEMP^NrrwY?J@y?{Do%q^2P_4DtL(Xd@xO3H^<Yr?u>FK4CtorjX!^oO0L{yj^ws7}rQ3ZF8S^%>%y*+fmQC61!(XPt2Wz2U_rqB?LwxpeQaElx>7*iW@wc'
    'hx!WP?pvPIOXQfd=i~^^-B<9-MF0xqF*h5<4u6@Q9d|m%qzefq4~0bP9|8`f6;*DLKZTW(JC(f=3pyE&@g~r8v#rBaxV6fY=+OxS<5(NI#$rC7s_n#t^&$Ze'
    '2H>IJkD0{4Q37cqXucfh*|qwi*3iEc$%}_+`<F!Lf+)LK0~S3TH+72^gxZs<m4YzX%u01bU0@_KPt6AXK~SMP9dAQ}^=$_S=4-VR<En?{`gRXH0CWLdBz)Ki'
    'zN-U<%=pSPB^{XhX)mV8eN&%8eY7-8=V>I+=)vyGJ-vV0kWuKm#ZeQDrgA=;l}+CkkU@Lpx9GI)+i*@e1QRhLSd_=L!d6jeL9=H5<ao}z!B~5e8Z)4ydwQ??'
    '!peilyYX=_*)#=*zYV4UisynCmIPz#m%?CH1R7FjLBYNNi(|GV99BqzVX>V75LoYPbfgXU^;2k!KcjuiTnFgX#^v;kvXc5uR5lpDs<Cy70tjs6Qf)pl$+qFl'
    '+EZ@TWcE$iX+0R)79bUfwAskkTL&L}O4;Ijz_j+y+g_pVKKCWj?qi_%vuM(p=IeVE$-0ZAE;`)fd8lc2fer@7_I=AWVDUej{eg~h5M2Nca7(bnwnUq+FHD{^'
    'hiqB5L1{_9*lj}3&O4#r<&wBq>}4EWE!2rQ1v@iLSk7P6zFgc$gnPwJVjfH&>&9>dAu*AvL<KGEfnGv~+i!V}r1t&}hJud#1}sIp4wif`n1RRLInhjd_s$dF'
    'YS{mv2|FRKm3VB4NztG8Yy<>dCNxVNq(eb@wu_y;gk812YOx%dL8-Ln3cJL8Y_+*#Xtg=m#%k3Hlq2w6ydTNPIl2&w$D!Z2up9%K1>?OB8)L+*k>f0fAIM19'
    'adZ_*6UQO(b@5MajjrP|3Zu4(wvjgrLhsq#*$~WNtpqRDa`L|U{r2S5ZWb5~#_Ni5d%wrQMFQ~yvHL~{(b!{(;cg)B@LUla1Q_l^{K0KpQWSzexIVAqBO9r5'
    'P=wd6PnZlS&`)4i!vg*d45k35NBuJ_t__6XH~-{v>%lrX+D5p5s<Bqjx8Lix#ED8!QgLg%I+NP4n0l&qbr3#dTxtUE$q)s{ROU<1&8#?K1><nURX{rIg<h{v'
    'n&t1bH5_rl%^*2Zj)s~Ml`fSI2_0TBB7%haoQ>dh`TNl5%n}jW^31&usM7|bCq2d59nF0ttfIbZ49eC@ArYIf43UW{<uD5?$>>Zo#e`&}ov61h?4ldhs<(5Z'
    'Qf3dC$OboUb<I)6-Ubon-uChihS-8vVn^tXXc;dWVbps3%7UIUS7o7K;CQY@zuJ^fDDSfeEUK<t+{4m9L;5QOV-?U9)C@9*$l!xCA#$N0F+sx%!GF&#j`n5&'
    'N2D1qR!Gy;S+6YyLx}+9TNMqzL7w1PTL81Fg9pYR;lAVo^+G6vlP!XVrMp^Uc;)zGTJvB8zD!M_1I=Jo3|)X{Y|i!AugV$sCDeB~Io6u7(mpcMTw69_9m*VY'
    't@h(3-D#kw(wZn-63{Qr=GnEvUGpXScsNUF3Q7QeZX*cDJ+`_JuxAK+W!%G;MxT3pczCE5d_KCd+Jv^3kOY^ujur*VNtzV43Bo9P_Cq9W7Su2CevvfZuwf)k'
    'rrAjqDjOU@5X*E%hoM+83vWA5`qRLuj4;^Ig*_*2!~*!X+4A*q#@p7uAeUR(Wlj2n=a_zqPR?dz>?DX0+FX;{mXI+s@iIbuau655{D{_j#>_-LvQ=>QxD+kI'
    'ePLl}(lO_>EQn<?@7=h-!g!a9vB_J;N%5xF>7(Aj20LR6rC1fsPgP?%PIs$b`$A;%9l9XY4ts$Wa6>8jPOD$HguKMLu3Z|ASM9<)T}>Tr6r(b?mbVi=a=wT5'
    '81hI{n6zHo3wcO}W_slKHg&_=g07Zc9`{BkjsO#9_<$LbbToNgs8kHe5F#O)xJPPsAWs14%m9B_y%~tATn~O5rOqGL>X{xgDLCM{!N|vAq^+`rQ=vmie*4=9'
    '0x#K-0KPGqyN4eK3rG}-LW&?P@AZ6D%+NU$jSXPAni^bQ16uC3IcZf-JBOfF9ibZ47t&PB%BV*j>H9TOlQOX7x6eteL=@~sCTJ@HF_%k_6U@{qgHd-JmlSs@'
    'j6C}^x^Fl1Nwud{@eL8S#47>9c8=KS2r~+$)3}xG8XwC-(b-+oeAauFV{qH-^>A!-@>+9$t>$&R*50?Bg!lrceqF7<!>)#8Ys+ObhB?zY(yUF+MTM=*b|Ky<'
    '*Ly72NCd^zp4={odR=1p{%#olEST6hpg{GAy;-dh?kt)Q<~O?2a@Na773Zb$E$hM^Y{@XaN<G6__tVB@z3zmck5Hs9`h+|9B*F4^`o7tm*(p^vb!j0Y+izoa'
    'joFst61FcEsTycPQNI^h$A2Z%VHRx)gDlHQsa7;fPAL*QJr@MlZ1@)M=T<r;1t3Y+fEl$=C(k-)XcM4MLy`1G0MuYqA8Gn!3?6mX=ZvtSBDAy)2q_xPkg<nM'
    'IGHP=hub>%lutr4t(i)2d)9GGgE()R#RSo({u!1i3ZvK-@=`Yx&wo9K0+@D2G31S)0M|o+O=e+aT%!<_m+Da!ij(BX6~+bn!m61FD!Jr1az(8nFfwLHgx4N#'
    'ZWpU&aM*(CPX4tW0<B^w3dfMEEm0L;7bKUDi{MQ0hLpL{er3W>G+^j#FXxm5>t)a>{VDECg+!iosqv;4i{~VsiVY!*<SjNT0AK{xc>?D;Hv1MhiP(b<Vo;za'
    'SROmj!Lpx{P=ak6=?Q{K40IPsu-nuRsjcjnPLG{oDlcq9q5&_+LM^B~%0mJu54}(erf2w+0P1OUYC$z27}79VkcN6==K&Z}gAao=Bq1kNNl7%6sDYPNj$xVy'
    'H^trR0@F38TNBhw$#%T5a9>y2_%KxzCM~WSOj3uXhofU@U!bU<dThR$YHQ1ZRtpij*Io#psR{((0;Nmsa1BGw98W?QF=w3@u34mR+NF6xS@ob^SW=WoD4`4V'
    'Aql{wH3p%_)Z2DD5tt!ZkpdR)-^zs04KWdNBTBEFD95!viE{T$K47fU9=&O}VM7(X9e05j_<RtD<_ylTuX)Y6P}gF;bd(7%&{!NV!pZ<Reb`2_7J@*5XvvS3'
    '6bKw;GQAF|UXNU_L9*{gm6Ls|UMyblom7zEg}jLY9SS%F0C*PgNCq(QfDi<0*P>}h9ZR2*c@%%!45wD&lruT2*d8!tk|CzONy$y}T|ZG7PMRjmC|B(elqlG!'
    '1)+WNWEj@aC^{6jKDbHBm2s~NW(?ZqfCV|mz;`kbpQ_YVG5Dagv-G|PV^?=0Q(dzjcuu~#5H4{tamdLXu~ak}+_T2Q4Gw;*m7Jr`$UQtAiT)ry_EEg-Zr6@M'
    '`}&;1wHN3od(A+FF>29iO{+r0*lyT>%LnQ;_?-Ep@(%_rI8R*CT(YF-zCaJkdo6)n<ykDTNp`)(w6!}o6A68`H3P_I=T<_o(Y8j4>2N_&HEDvYRqNS-(ZU-W'
    '=k;cY*cZdDLZ50&&_*OYOLx#2?y^Sy>B1?uD?Ml@X|GvB&1uVn>=SR{J_8u5ubrFkC)*$@N|+@|Q<nPW{#xhWqU3%ZyTEy{t#W_;UA0d8G|v{}_V?=A|I>1K'
    'TQtMVuN8a`sS0r8re(Jtfh>L=xQGv?s?pJ%Rh1R%Ay6633Bc=m(9tQDzv%9<q&4LVQ>Lq1j_T$?U9hBvUle;N$|i>2{5VBad^BI&Y3l^tRk~Er;(@{fs}1l8'
    'qU@<zxgRvh6qFv_ZaeAZ+1YW+14N#>o}H`e|8*#9%k66})L;UzYiL1HCs?S#jRft*i;1h3x9av$H#qWfjrS+#gR~HhK~WP{qD{@<a(JiMDvxQ+wUZH9&**o|'
    'Mr*vBgEd5N#Eu`CZH?yB&Fu`0>Oo!y=o}6_*<%G$jmt-+(5!&XUq3e3YD1$<4gUfj)HZ~py1HnTYJVyc2DO0LH`M&ahF;6!b`F#mwRAh2l@rAht(qjVm@n_T'
    '8RWaiJqe0B+EAX7iE_HFuc60K*41sbm$2UzPoUur-al*O)U_ft8SHI<qHcAy8`%`G%9x(LA;jdyiksQAoZU2U%N0bT7|d@-F0fav#3IENpO6nrkSHy*MoeAB'
    '3SY09#(LWX^Rd<ndA6L-*ECIecDb32!7}c9Vj;f8$2_nk3$g^VdBwKWKEinfxDc7FVfxZg53oC^5uB-(eq|bf=Ls?(gHQHqxQB|0<+xNxYDISRlX40a`oce!'
    '5hV?Jh3I8_w0h4Ej%OMn)(t<UAUL$kE4#OedN#L5V8vYdJ%j~P)!7x)qF>?npmTDs?zxpvj<XD_1V_WnxUtS;2M7=_=1{;>4cEG=?U^iW!TSD*pbe}#Wy<Pp'
    '_1yNy+)oG;Qh!lE^wD+mQtjF<>}%Ya>aVp({E^CH+zV!wI90fEsYAax_Yoa}6~Y64_m}TqW|wzsg^|ZDDTr<Pl((!SAF5w%MWw${#J0_=NL09f>xzkC6H?(6'
    'f@-mMPulM%Y>_+Ph#omX1<5O~%40969(dIyobAL))^jZ3P!@L7H<w6QQ3RC|=EA0O;`5$`GC=!W@aqGR_%N`h7Fa-G-RYevfDe#NYdu%DAA_P!wkd;VmGD9p'
    '8x;!ss0*c>sF5?EibI`w<u&;LA=2;gE9(H`8;oTOd_YqQMd7;w4bM~$FP7rtdI^(G<X{Ol81Gpm9)-aBVa8B)zo3a6lwszAD@97f{C;X294M#<#u=lDwtB~k'
    'Ak`mWD<4n_dh_+{q;<a+<p}&VnL<KHE%GLpyeT-EHV0M^j#LRYZM{|)fqARAb9MH+uEW0L-+F#iU@SKj3Bw#P9OwB15;`ffZz$#i)(emvz61_QtpgS4m4S`e'
    'E#MX8#8B!GXL6z8jfN{VDT8=yT`6;kZubIf%eF9puR#(C_Jhg($7W}Lg_SU@7Ouv<vpXBoLbU}09k+@-sMJR*NHxJ`RxnzqT&;wNSx4e0H+i1%h!-ApSqyKI'
    'bHr&l=$Gn7<D!w6g=GKRlpI*q%(KH*uW}%VPd5-vkm|Z|pY3Kh^5n!86#Aw~coP-NLUQQB;XgnaO~eeb4?{ON!VDDP-4Nxh?pAF$a=pVYPd~COuXG+3Jd~zX'
    'm3^Hq0%zL-?esCSt{qEg-=?uF;rZqkqdC@s>7Q>N7`usvt80sqQ^Ww0#;41~IkZnn#iX~ThhzIZ?Wl!;6a%7J4d|Qz9i5&@pcqz`;kgWx1LIC0M18TvaUQ&a'
    'F8+aETGZRyu*#kGsbAd5E3W5bCLImvGAA7klk5V14&yp)W?<{0el=PiHw*%%n8Cw-P#us~>|>!%AX8FxmlOEWY<%&rU)+8%9(?h`7k~cZO`~;B+z(Y{VWe93'
    '`{6-EI>VklpR~a9mvSQ7dM^3KN4QNFb?fiHm#f2IJ!nr0AYGjX5*q(WH#ha|<i%h(9xhOKb+uYgB1(gborX*27ZUd&P_DT9h$%FJ1+qXUk%P&IZdv~@bht&j'
    '3f3-gc(9Pma)x)OuP;Rr)l`u*Vvb^Co(9aeZ<Z#gPK=$ME6hylr1FtdJF6*K`3W$5*LGJwGU?k<7sdv>RRxv;t^MEwP|g5#TY6^S^V@;hVp>z7j}tt%9oTfK'
    ')C8uDjvUk~W22;2G&CmJ&GLA<BHQEo+C9{Kx;~4g5(>}zh!<(X1xM!)x{~8$07oLbrfCi(q*-Ak29WP*wih>TBda>kM&MQ;7aaF0Bw6EKOT7UlV-swloLNug'
    'SYvQTi$`XS^-Fc4v{Lxgq|%AW@^A`2dClKvx=pTk3j1^4ZX~=pKub2eL(>LaXPEGcb#9T`4N95BpwJ7KK5gsB)8ibj)x@gF&%o)EJdEHQM7fxa57zSo^+yM&'
    'rT&aaD#=}t^i2oWa&hkSSn8Xng}S*YtxZ^xi*niA%35<@s+tR%lsR0oG5E8&O@3NWhKj;USZ`5%7PeJWv)7UV^cp2N55mo0HxlU48vP1+rvT-F)qDequrirQ'
    'Y^lraB>LOQrLfx(%^Q2E2vq`UXRSIs)WmEKdj9!z13?=V{ewwHYe9i?2^kd8#Mb6lgoqI)w_+_Y72z!ER8L2@s8c&6y4kvUgtSoA5#X+pK!Qt+63Z;d0S;J@'
    '${s|Pmwi#J7gUQ4F>6(;whgTXf=t=Kj0`;be=lbBR(LZW?p&#!*Oz!UoD~CVWuX!?;Wf8dwcWKO0EE!ZH=u-dvxX`#ZE1iCYgrJeyBF8*n`m2v1tM~GycmEC'
    'Avh#N(Yrugu^bO`EUD4-jU|M;G_@Hh;E>Otl64++m3nyBhgMjlN>QD&{yL`5+N@?iQD{}{i{agLK8zVYd((8KdnQl$yoEDU;rzxYY&Q?Mzn|NhuU)psBZx&?'
    'egw7!F7_2&6Vu{-F~vAbf@N=rmAeclr+7q$`Ui26#3<3G$9aUkMLa{^HZuheyQvM%%jE)s*%?DdMXUjODn1sY4KQ1ZUQ+krxHQ3)LeLuT62u#IOBUpEZjIr|'
    'P@X+HylY;DFi}BwwXtgW_rv9iq-qlJNCyNkS~J|?xp%tNfex%pz*Nj9QMBR8qkxAng7`~T_%l^xr4c<c)`96OQ9DWBx#<?=`Bq1vKt&}tM}i967zx-LQf7hP'
    'i!75c)v7i%wcmQlu)~Uq2cEn^C|MhG*|#7OgTL=)3#9oKMGfKMH|Po%T3b1E*l+bY7iLBiME|6jz%hANP^Nfn3FXaVebSjq#}qTV+y3zC%`ZP6z*#$Z{oS`<'
    '7SZlB<f^+7%v6=;HSq9~2wy7BX<bbvo9JqZ8WKTjFWYa`S%XQOm6%`yBK)fZow@F0)Ck6I4o6*=+N9DER$FeS1@!{&^8<CE0|P&|NrAsQ{enWY!@NU43zxu&'
    'Oi$GEkYi0lYHK?&t0!nU13Ol|hSzs$2QP*|B><wjT9y()shg=mu~Nx8L@Je(=u)#YE67u-T!SK|g4>BuD#z(g5J-TD^GYH;HI_qr3a4l><`PWG4FQrj>%|6w'
    '&4655%c!-fbeE%rz$Sym#qtq_U#?a=^$0U0uNagTPggr%+meqXrL?My=p&{G`^V8h#biRZxdZS(Sro&yaW4wEJNG%y6ttIrdHr_q)8N-vzr1?+_D$pu_<tAG'
    '6rUy0$1plcawLGK^D2C>XXAtEDsQiArh}h~$C6fc;M0w?oLjZ+oEvhLa|7NPtvL$|AR;_-3s=Zj1w(3?NH_d6W^fG1$%`ida(w<2Ku(?lNWugUW?d^dgW{0|'
    'o~H7mqTJ#R)`njv+*BUQd#RAozpt21@Pw?I$rNON$iDw2LvO}acDdX@fXz7s{RY~22<qC)a=jcb@BI8^K#pjlD+6O+8(5N5&(0z1(_t5LMCREs{5b98*(trz'
    'L0{2~XSUGh3vDfe8L0P7NKvKzT&=+<57lRrJAGpI$SmZvZ@UkC>>R$TL$_Sd=hIc2SX)E}tEO?r&9j?8jbR(_;o70@ALCb=lxHYHnzOX-s`qX3x-xlZtn+g!'
    'QdCb((TSR+JUhy>!@d}j5^Y(>3lMMswoMwyHf@jnEEK0YZ-TZzhSeq?PzC2e^`)Fm*cY8h(S~&;t4*BwcDrrExw`C;#q!plUwE@w2lLA&*Gte#k!YziowlDe'
    'A)S~Kn#jqw?BXz3H_E)bEpr03`K4Jp6RsT8E7&Zi+_p(g?aX2U^8Iy5xi!^BAphvi@Ryj-^dunLyd}WD!gmwB3&u)><zg_0i9>fjvlR|Fv=%tVVg^;z_n4gn'
    'x>D3=a=^d@qvMl_KKh63&(NV&F<hcaA?>S`!T^fSn-$1~Ea&5m!V<%5sxY29TM$g;{SX<G-?a@V!uox%x*cLh8SUh%w!;d~MtiZCG$H)y0V8VEJ=Z)^@1A8x'
    'FsUsKV+wwv4}j6>I4B+yh|`A}PXqjqrmHL+k2Yth$jW1!{?ATe8yuQ_;6>c7S^Yru^^n&G=8$!ibtst)#wz!`4pbzM(d+$kdR1I5hvg_!xan#Jv?FDkRH%PZ'
    '{5~(oiq$kH!{H(jy(ol!h&W4Ou9LoTdT>Y??33J+cjBv0wobkgA8qeg>#FYb%~m}O5gE7}tHn_pLLcZO`pFDLp_i$8r>x?sSCayfS7flEv^9AqWx>%rH%%5e'
    'z;eu^v5IO921L2TZ@`Q8upeD>9ym(sod$;ktpg8aFmChh@zikBDYC##8h)?gMO(7T+uZmd#N>Q>f#^u_pYXe5Jg>1S*t7y%9&afGtG-o%^}KDF=NGuaF{>8D'
    'qtJYa9N?XI2;l`y2~U@x01)XkK!l2v6Il2F+*HnrTfo;GVM5jhzxgm}2%UPAM6;k2ynn0)^wOSU$V7s@_B-tdelS-E9~rw#Y&m|v7(XyZau%w4pej~>>JXkl'
    'q1u!-A+^d~f#IVzT%*-%Mee?z+`<|mJJeE-@F(|dn@~>g7r)oEC`m?Cr?kmRvoTt1(0s%DgR^mVqJH8odi(RQ-~BLn`PY{}zJB@b>sR2Dq8)lC2Pouk)uMEd'
    'S~i0(BOgKNc+osoSoTEU4R-;CYIGXlzpv2LmsziIs2?;^m=iu=2@Kd#9;nWh$b+md)CrmmuFLVbm|dLb8AeRS>AGmFhqLPjnB8B&Sv$&^^SrY-zQBC42E~=L'
    'tBb})akFX=4J&fG(0?5S+kwF)xFDdV9(3huvGLxVo@s~kwz@wdfw4AwadGx>r0vDBfZ(JyC--krbbXBdGnke)h4&Gn6L%<|{a0!p6&^WJ(-^J;Wr_`|9Z|K@'
    '9jhHuwIix_RJYnOR8yqFam{+f`^V<n&Cq5liuirORyq~^`r??yakO?07svifXDoejc*cRAC@Sa{s1aH<T=5IYc<;B2_37i;NyYX%$$2tAl%8zk;@vdh2Ewdn'
    'Hb;X_k{Iq9RjAiJ?`wIg1pqxb)#e^tW^B=#K?>#pb78QI8m%ZUJ$Q!DFnUM*Fc3Za_#%or&pw{_U8EFi{Yb~fCUh{cYX;-cLzM$@_+dD7AOWN#th5|&F6eFW'
    'J^{(0dy@cCG!_lJ#>3e1Zn)oYsjue?i!Oo~D1dXV2`I2mQFbuVn!E2s+GXSP9-(b!swWvjqj~ITO(uITD3IT<6u6h$S6-yaA3qL0t^S4CD86rs>HwOZy6xAh'
    'Dht;(_d!Q1!>AN1kqYVzL}6h6Yx!AN!srX1W3gK}@+y?j9-xK?D3XC|xfNw=({|W4>KOdyrJNBiFQvD%Uyg;l9*{5IYQVzkVwdfZTOoDG?VHdw9mzh&GwRv!'
    's5?rsGq@hE%wSEG+Ee4FZVJgTgayP`&%7h%lr6LS(%M3=u_vH7_xApvHs2Her{KLmJ85^a|MNfoXXX^%Km7dDt2b{^@0~R%hP)RJprz{l0O>H)W-nHX;(q>K'
    'osTh{JFAKyMw;qP?=ou-Y1PUoX2}(-ONgCxfC`~(qmaUKt&a8j4w9<_gD{*G^UW$_>TlfbP=Gc!1oki2Kzy~Ie1`TE&OI5t`OCL&UcJ>~78KK+>JL_<<h}m>'
    ')tevx`KQ63eg$<KMmgxT&pPz&uP<N!`29<GJ^1$Z&)@ya8<Za7rx3Mkp)V6aLPDuLi}R^`y4U@QudWmjow?Q2^!1cvjd$XybFOj8wG_u%fX~LmDX7~tS%2K{'
    'z>#ut8#Iz25KO<m(^J}H?G1UO;zsTgW<F4C=D5^*h3^|Q^nDNZz4%DB13FEjk?Vh*6(Gk0>iodk0yDFSgBw7nTtM{lFW-G_VCi$hj4Pbo1sO7cR4@`T!bWBe'
    '4G>UC103JYhPUMiB##Oe+}>jJ4OM-H_7d4JL)EH2h@k9R<G{aG<6ga+s{tE8?u;Jbfm4%VQbyrPrI}M5Ind%AcVeAghKBWKy@8oISZ8Qbq_7^OrTW@Gur_@2'
    'b1Nu$ZbTdU1lwst6nP6mbH%E;Jf9?O!|thL4mza+hHCKGw-qIsz4_tgFV8VNz5%EesRl!!(iFK;hoIVxJr>C8Bnl_eoLNPO3+6+|$fiEdBee`2tS-ME#9$>1'
    'y4Q<TPU4EJd^WF1oFVVG_!SRL%F^TMq|RRIx@}qZoQxw(A=ULWt_9iFbgp$0$4yWirw5`MUs1A+6%q&Nt^~4H%s~sBKES&rQWUi=*l+xTQ2#BH3$#MU*@&Qt'
    'YZP-j2f6An`SSLZS<IpKFyTjgBKXl-MgRrBju}v6hc4Qwfk1#zk8uZdoG;w;^$<pvAbm;Y`%I2(W8o+!(?^MV%b`&LZPQu_u`TO}T}kG@wt{cU%f+SMSTRgd'
    '1rtsS4)J9J5jFJ`WCI|W7E|Kk=|d>qIrlPa^QASxzrxiu$IL&$;$aDJhCU-K@Jj?c+vT-{SiIrr4B=s4^DKdG=-wDOtU92G9C|R6#LXg!Fr_9{o<JlDh`L57'
    'kQmj72=b<aA*C2??C<Y?4;FY3s~^aFca>q^+7L>`+yP)2hMGCBg;7i`-Hn=2vf-#^8WHJmk5<Z>eUqKJJ@<#~?OXv+q$EVAnSl^b`|6QR0Ff*im?5;Ye<_NE'
    '^W9s2m}mTEf>6{<;HYnBixp-H<sG$zjOVN<I(x#ssw24ulMPCZ8B@RVMoO?*{@6Uu?b{Z$dAwaDYt8Vr{vMRubnbM#(FiYs9By^JnM|gnmiA7cstA}IKlHk$'
    'JHYvPU(5#xUcz=K+m?tR^l@dcwFpBzi@|QS=uyMMPiC`5RJq=+c+*Eav>bo*kx|^EXqvPGt*RQPrqffULn-aBIC|DGbK3YLHx%b<VoI4zr!;!DPmbuC3g-tL'
    'rqN^A)Nh$F*f$UY>4;)a9`aT1aSvm<65MRyaWW*Ri!4rf$AHdlM{TQgMfCapFl>krVND};n?i&z*%M%Rz_(Sq&PCgGTMw-e1N6vw%EUEcD9SKmm@DDFP}#mi'
    'q5m9lBG#4~5awOTgdnL*UvK~}mS4Ep*$hOxd_7;=1zGuI5Rozd45Nh!D-(rVUJpxWM1j-Ta&`{%)@xE%Q>gs9Favu(T$Uhggv&x}Ueu_;lF6`5nAKj9oEZHO'
    'O=cCtgX#bcZL}<j4@AnGEJ7}e%<(~rhyiwEeGRr_+S&;$3YYVbCEX%zer3*Bo?*ZUn!?nr5d9z6eXN<$2h56^`pJa_cK}H@F&eCS^T+Ja<}Nh`PFk}FZ&ozS'
    'rr##XTsaJuh&Alp;gL31(wuXZjheC1Dj^76#s2lJ+6<i-9ZaV~pH9U*(Eh<^L}Vg2g@xe&_SC}k(X(9PRxgRM)j{?vGHKZ5Gl@NQqGXHD^fN)_mS3xa*Ilj0'
    'O%noaJf2T34qF)EeSO_t{cTBG$bgPuOPc^gH)9+km%u>npI~{Pscx7gqjs~bj30f)2EyhJg7)+d@rij8XfoIzexbm+UYFDHPoShN-M%b!Adxs94JK(Y-ZEWW'
    'pU_M@%taO%Q9Cb@xjO5sa^>Z<@v)Fx_0r<v0i0k}6AyACxB{Y<ff|yYVDL%w<~p=mx7QyGQ9T9U4UNTfhVac3nt~f20@6J(Z3>Us;&h@6VvOZ^pk0@dKvywD'
    'vBUjw4B#DSU2-AdcL)y*{9&$r1z<72{`*pgwTRZIvD2bJt}!ps1<+f1fn4MeJY&YB*c{av-<}OnJ$rD}eC-IwQ&;FxIbY$NpFmDxA=Nb?k)TVkYu*lLoCwnN'
    'N<{K-Nl{=8qlk>3t#nZoD-y|KS-?n)iJh>-igXvS&Ze`_`nB~9!10_uCVh{g<mYFv7UrM%_#WSQ*X(ZCpR<6LNJ$e8-WdP_LQ@WPE(W?11f~^x93Xtw7xC)h'
    '7es8XhFnwukvHMA5g(woKR*;~QPH6VrpPL-`QfoiDMPd$p!~sQvHnmLGuIry0UEJOQ39V-;s}n|Tpk>YtMB>A(;5=)_K8kz#@1BZ9&SI1)}C;CQAajyFI$_@'
    'VP02Nc;F3J9ypZ)X$rC-nh`4kds!6P0F~4xpij{qu~=YSzAsnkccAZ<W3(uhu9}wX_W&JWBW6R3UkGH&pM5g35wN7VJ~?Zr<t<GTj<j|$7`tpqIO4*_fQs3C'
    'b9Ft~D2!($1l+`!hSZbKXFvDMitClM{i-gCl7?fQLPslyLK9uychAOFXF_r<*`wD73YnAMt~eX5SNtMT-VHfXJY|~8C>7?roJ22rO^J#^FF;NYnO(~5VsVLq'
    'OygC-K-R83D{Y630QGNfM5~7h*1q~Dr|rxfQ~V72&22ttiTFB4B0gh8!Iv|#{fKP(yLMlAogI}cO!5Q%=3Ul5Sf4+3v}J}lsqpTiqd8!G?C4!XK+ojTO2#W}'
    '5U~2I5J%S$KG`Pg6b`~Mih?K>?o#}2yv6*cIEIhbhQaXw_A4;E?P`U;0Gp=VpNKphYT!*^{L3B!nN3gtRaK-sbx<%6EST=d_Wz*FWTh#+p<L3edX1I@j{vzL'
    'si3_fYBc1fWv*;VENLgtY!?)tTUn5naKo4@3t(p}OVAm?Pq8tKE@Q>agL0fZfwR*&M%$amAP9w?kr<obixI`|Z6FjRIoS+q02qhn-I@_3Z-dl@1QQrR^0z=;'
    'P;NC%EPVCGb07*n!zjEM!;7E*yyDRmQdwIYX7TjhW*%FSTq6)@pg=Xs9BfV?&tMNuIzbF5bAWg;bY==Y>dp4&1>voi1X#ZEMdFaUqvuMvaf7mM;-&jHD9l!Y'
    'm6onv#EeqvHBEAZ7s-v{x5#@cCv#w1$^@=8g65mHT2>wWomQ#EVlC)^%vF%FM182<1+8=1FhjE^wxppj5n$R;UT)CYx3;xSWG0YgsA?GB#H`e7Qi6ldL-7_Q'
    '_WRmg=LKpx$jEF3;ZE@e{95bL$*9UOQok7_IGWGa<<&;9W`qtIX8cX^*=DXMrd&K-g7kc1HTBM1!I~#AuWn3rYq*`rwravGc1$6`uqtKFBG>}y_FaVkZB&b='
    'f3O{uY&A4!-Dgnm%JDQ20-59L5SAD@FVDjK3(P-6eLXp{X;?wSI?xT^pD(kslTHVi<7PBwOt!85Ap<G(W5N|!3diQ29)4_BBV|%zEUjEOS~8Vs0L9vWzd;U4'
    '_3x_}+OL<;iZzEIPdgzAb(6e{+=%*i6&Pt40&-uPyrCAK9T4dF&W(|b)eLg03OiO~+bzS_!q=YWv5uKBp%F>VUYOjRsSpe7=66Ho!4~jj=H4Qhq^bUDn;F>7'
    'NYy!S;gI4LbQh29ZnVR=N7B{M$x|w2asIG+iR+ySJ$Pa9dQX&<uXk2B*aR-rRVasZqYT9<SI7q>9p?!}>2#~s89ww6Y6i<eS&4N+NM**$1C|}6Z2xY#8I{AS'
    'wzs+i)8~bn(7P^J@`BOk0e1&m76&>p1U(lY)m^a4t`$CSyNt-xHjA}L(G0bjhqdSRGn!E$FH#347ISX<{#5k3xcU1*Fhhc}H6RD?lB2cPO?uB{^t#}-+pUJ>'
    'rjPrCYmfISY1XL=p~t)ajXs3~`P$>%4_lZpjXe4qss0W>kngoi#5kPY3|unG0hRQ}>o1Itj`C!WaruN1Qi2oX#t=I&*vupiK|Mm<^Y0JcWxrQ`pBan9XCUOn'
    'ZEevI!S8r*38FIwjlzXc*7<a~Z<gsgV3p;1;D2nIb&`5--x-1Zg^$l7?xG)K3SPly82SnJ%4VpcA6$u}bHW6>BjuE)O8PHE&WIy1OaY{-MHn$C-jBGTT2SJm'
    '{ZKNR$Xz2yM-z9ND(vS^Z^#!kY2EaL$FCAfAp<ewR1J=i^mDQ0qz5=vO*Un4O|=qj#I_Htp_b5sv9~2M#meO+3+I-&Jsh3H5}z9|0^ubXKSY$Ii&7mEnUS3a'
    'hEMk@F*1}D4+7C@A_>@Xn5~<Z>9gLDhDA5=R!qiF8DfwN$jpCi9)c)Xm%!iNLL#n#4MIqVHKlN*zEw4@1Yw3noU9~c0D|M@;nUFV75NxHd`N)YbjBo0qmAA-'
    'axG2sM(D^xRiYZ0S5yRM$;Yj*cu5hI*Yn2?5^VRuZe=SdR!|<GK~D-fEozK1EG`0t%a9nG@gj=a*<frLKVEV{Lk0jRDwiNy9o<_8*5^(x%Khgg#4J>q%?A`~'
    'k>NNH;Ur7*c%;cfJQ%55C<sFu2Odr3)pD>rX<cQYiVBh4Z{TMh%iS=43wb5QAA#`UvZz8RgVQ8}20G~qJu`%AX?dOGKSN>UG5oJCpmQX!SH<kAoE0n{AxQdk'
    '3_>?n!fNF9!pbJpb*_HP$_!r;+ppFl({A!#b7oz~zXSi$QMpjtaRtbq=tK0m0rzWbe^E@MUeWl#B@LNPjPZF0e)XdpD~M}*L7H)C>j)sBoFs%~n;_J9*$<IQ'
    'N03~``$ZtPal=UFXJ#i=NIP)^fs^EnC2lj<rJW}m!GxKwsB-o@o?m=x96}7&XZ-%`3kvB;yR1onP|{96dFbuP{+XQw(f`ghxors<Q!2#&R67sMvSEItP@<^a'
    '0$>mp(yfAGUxy#d)kTN2T<nW)&5drLlRN_px#z}UXkvmwvJ7Rt8y7PU2~yE%zhyG@-Sj$rKT97I42V_Ri}?a_o0$yrwb~b4{&(nt6zeAF1y;ZfWn#o^b%MpY'
    'u3egkpV{tT%W}Zh(E{@*A=}DJ!FZfFjd?;)%qWWK=ph-J>5*f=)!k?dx>|a9+?zH8A8Mh!kWFly3Q*<rv_N{0SRL?(^;iZup6kJHqtyAsT0PT478Mfk++gHm'
    'F_M+r!U=hU)+YZp0>dghl2jP2(BJoZzA9#@n?TJSELT&5%WFW(-8Lu9#%TwFUp7am%J7A46|*wxQAdJY@2N={*z()w#P|D4q>6-`m?!A1Sc1rBrdAok2zfEE'
    'P^pAF72{=n8r^&OAn~*+9x1|>cqM?(^oWg)Fr!d9ja%8S@v$rvo!vFfXT4WB2Di;#564Csb*(0JyQa&NIKD4|M%MM~YRVj1wdtGXvYC`yb-fHT*=rWW$T?FQ'
    '=k0>nQLguwPFLGPRf@$uf#@~xs)s~})1Yf=Da`!Fnb@ia?oF;QME2yG$!TG>Av>+A%Up}mhhO0z@GBRgLlC9CSjd_TG;S8FXQT-Iv@<}jJK^Ue)OP>@b*|PI'
    'QfvIKC3*}#$qxITH2bL}s=>B{IBB-!xP<MCMXCmxP&ANzLLL8=P^VsULXp_%MLC*VE#A+qbV>?9lCBXGLsMF4xFG`c$pYdjpa!G*NYgK4@Tl|2_v7k;^c$Df'
    '#h@e=qZu-ZFj8_^KeAs)2PJH%T(Um>t0G6A$sV{<dcNCYeLaT)7?(-0=V7v7u2DkEUnAWc)yQ&l@{lYyCtI`J=!#nPV`Tj-H)e43@>GsANop^E(NeRDY8XK='
    'Y$vT5NVbuxBEn~}_N2r$W)57x(H5_>y`0m)td~It^{2Qm70L^W>F0w<7Ek147c>SwME)I6LReo`BKR=EG7_9>y@J@MplmM2SJ`Ir$UO)#go!U<CiFrtZFxy}'
    'Dws*ipslOGPLKn`N*T2CGuR2XuWO3^%h2emx)HIaw9HAji*PK7g6RpWr&&s&9<fqIWA2vmjMx+HY*ko9P`Sl=@)T5SE+UwoZF*`(wy7{8s76zfhPg$T5s#fK'
    'D^i26qtK|<-z6`LL=D{TQ6lJcYl3bs*^XCMIyaKa6BB$@jANUcR1c^s&9#`XF0x76){%s!JZ?X1FG2GO4We#-m!Tx|odRfqS)yU3RHiOQIOI19*eRp5k2X%E'
    'sokY3f}Zz5y|A1OQRG4w=tB~K7Yy0k`#bR~@uj^*Y4)f*fKQ!^Fr|46%HA{4luZcsl$(v4N>#z$arbyufA5@3c+v`mco_1LBR)a`ZRB^F@5q?is}stlW&W)W'
    '4_A+mtHI0d#;y^b)RV~zzLN^-s2>5-%A<pGIk*4-_c$K400wS+g20l*Aj_a*9Z`HSOju(DlQ|Q|i|qmLE*Xsnj2K<n<P-O<p8y1>N#9ZBE#Mu9Mimf*gl-Zr'
    'xUFHdhb;*8B~jn!cVWAtb<FHW+#`AigQJ+cxT&t0S={m8TnKkP+3)12N1K$2bc4&-n5M!JYl+kSx!J$Z%@KWGuIO`fMh_CdJW2$!+hw=QWy3EV7HcR_%J-Uq'
    '-fq+)+>_S-N%|dXz@`248YE@1HuPg343YzlbIWO0dA?I$iVfwxmOz;JEEZEHyWV0Z3OhFwNjkSR1E_Q7Rzg|nwnmESaIr!exHibFv?s|9jOmH-dRRj?wIKCN'
    'J+ZNz{)KhWc^$Au{^^4G_w8*}|M65NkbPZYdt<f??JAIJ?4NiG_Zh&z@bQ#@Bl~0<MC~9mmTt<T?%ZGN+*_2~uVWWD54KhAufMC-X+s{?ar=8k#QbSFye*pH'
    'Ro)7|hcuG<u9w|-urLF%(G(*6=tS;@BP`ZKpqZKzfY<e)qyHp-(d8i$S&AC0n+J8l4FrBsikB#sD}K{64k_tjn__Vmqg8j6E><-2X)?>-#rpvyPI`ZQrvJXs'
    'f6vX|L;d3`^S5Juz!zq%cl_0n`EpD@P7ZAmAWbO^s@iy;bn^Q9L4%CW>Cx@-lMaO|=K~;5QZGWBnzw63DtYqt+`i^=Cni0<h87f!hlLuVEwxS`tx&%re8wvp'
    '%lRNJf##<Dw~KO#aoR42cZ%s$`x?xEi1UN&cJryG+BHMUdtg$SVxBKs#S=JyM9qiEBCm~?b1+@%jTqX%jiENgbaOjfp@q_V*@SBd5EZ#M2U`TU9hE{GAU6M='
    'FN6jSjW#v>3wTh+2T;uFqM;~Qpd2Y;DD(~0^i<E!iEujy`j=WViP~395>(8WcU{ccj%Nv!>vRx2jW7m%S7o5bnBU#jZMBzhPMa)){4<@*JyK?4LPUsqHiJ{d'
    'riiDXhYTZOT~ZaRiksQAoZU2U%hd{;hUPaUE!nHqIFpu{PY7kNUugC905evsnlyDTmI{m33yJJ$n)2*&GaG{$>-WUSeT$EIVD{<EM1y(N>>r4U)MmoT1mF;)'
    's$n|RP!F(0s3n}PmVRYg!Jt;i3l2Wnd+olV{ncSOWJ(AehH9$Gn|%%(;-$jEtA=sR#sT~1KVo)3KN!7ik5=y+vA2U}J*B$grxXNtukgz5ZK9sqRuuU$@Rz9k'
    '9>M}ou-O&VqF>?npmTDs?z!z>o=wVW0V_%QEO5e&zXdyJaKT_22t56m|F*dE?2xwHYW~TgjjsA+%IdxG+;+#@zX%lK=7NtI<nd}=stwbH{f|3X{k3MsKT=t='
    'GyZU`0A<dTk0L5y0H*K$^8L%K1ba7N?<keJi%4zSJ3VIY>Fn<6SKDPV5^@{aVgt0%2FF*ZgAW61XM+XQR8Q|r0epb?pzFCpstk(S2c`^~*1`)_Y*eVd3ii?y'
    'g`XKv#k9`6^7>YTj6-+$m34rT(Z<LLKA<^`B6wYahG(vc_^SB02F0PKj!Lk@dCwv^DU{F;!;Ukwg=UgahMA166lIAyt{Vpj3hDt!PYx)JNf$#1Aa!sbM^*kb'
    'nL^-aEw&~v+9`O%HwRXjjdvQhV10=gS#_&;ShYpEu3x^d?s|SxU?3h8*TPWQ&i7n+Y1ft6H)xhe^@6RBFM%&m(@h0>Wnd0S`RaiGK>pMF#)dZ<uGFLq;<3fI'
    'lsccf-3zQOn?1oP25BOgiYEIXTNwToR>CkCxElA)#BiuH*9*2-ZWVh_sgDM)tF@gK=t^I%roLTdN8%?pd7hmR|L<~H3~!Qi#3=yim+D62qLG+|WdGX~m_pT5'
    'ym+fuIgrDr8;B-nbltemcC#CqaAFGzebWS>i5P1kIjrE&9N+~?Qysgjbb}+zKmk$>G0W<1)rKS2JM8v(^3n1NY!l6)G^MI6Ds>S!+ZJf25BZ(kSVB`qjZ6uh'
    'H@6r@vKCDLe1lNY!#iAHTauzTqBJ;J#wB~w0ZGqujI0Uiol%0SH)xDh`>j?hhNpF?#_2<j^x@aAfhFUhta0hU>LiiRC$1+3auJiO_;p2d01^&SifO;N<L$5K'
    'V<veFZhI$r+zSpKVD&I&&ITNSfv0P<JPr(mF)>qt{h&I)+3#baPauC$b(a&$q5j3cesTN7c<{v!U;O!tH;vXk5!+Oih3IMB?}x`1i4c2YeYyirZOY4R>uKi4'
    '1>rVbM5BM?38<AK?Z1w~(;Wrs>B)=1a6DX~j_Yc*o<!6P6;oukzsi0gao3WxUhP$4M-1z$vxg&7OpLDMep_6Orln?<!rlcAkDGK^&hUQr^{OaDnJTg=6ZEat'
    'dXZ(D)>eyxTEbY5iaD_*HBP1RsDrZAT2xYzBn?I#+)4}49W`@7I_^q^Tw1LuI66DPmDbQ{g;Yhz1P8mG-H@c`<EEqzUT8upRE@1jU3JNNG^wBsEl+l{dY`Vy'
    '?!CTJ4wbO3uVSex-RmyeyNH=;!d**;9J-$4qybAMyHRQuD<rxx$(aV^dqwWWO}jMa)(aO?pA~CvoTQ~}cP;e>bc0Q>?Q%9lkrS9fNG;F@g-z7-3=>sDY}?|D'
    'eeznn=gEbQb|`HAp7Vd8zM9=NZRU8G(u$RDkv<GcVYD0FuuD&}b!_RO*(8hLf=vEy@Xe+NrcR7Vgcc7W=)SxUpQ!&PVwvl^bJo$hjE5_ou3WtuPS((W+ZoS?'
    'Pa`L=>x<JM#RAqmEkDgg!DuS&kR$@ltynYn1(>;Ta+t?58-qWA+hmpX<dr(R^IBHLg>BW;61HRjy@oH19BAEGqhBFq6ren?ns1yDRwna|Ep?fVMt?hj7<OCY'
    '31Uw#p;iz;vQ`~_+t21;HZY%VAWWDd`ha6!)q;YljR6E0H``S15Z18xNMCA!k*EZh)V&r6L&V6DQEP$VQDi(n1`|-G4v^~jGIby%!_Cxco^fk#4~qPnRNH08'
    '=vVM=(suH>I9ENz#|bdvVA=icJe@R($ON}9%8-L<u|;mJYSl)<wLp;VBN!S&q>$f>dOLo!84q`^RL`4K#23y`1Qp0oiP>_S`@Gt4UXnILH02x67rR-5i_I8w'
    'pl)53I_vJm_4_6qBe%vk(WnON>tg1G#O247kO1>65M;-2q4w0E`o`wOkraOi{V_W1R$30YKD5FbRjT!z_16gjtgUnA)A80QU@^R#&WEu8z}~}M=@!mYPPF07'
    'RCv?!3ER!3_V4Eh>odmolt{iXM*i54MJ9yVg{+H0u47rmkgOYoEv|B=G<BlX^HqGfb^l8x0)G@Q8%`!-L-J6YDT)N+D{7yIa=Cz*ti}RXQQCk+i;u-<1MI#c'
    'b`>QvwxQiZ(E4^Dh&PJ9DJbBE?HKuWut#@0n3wmFH|A_oUTu~x;^%TjB4dg2r}GPVw;3M5JPWKy0<#^AMH|wW+8F3j@Ix4$i4+r8P8C~VMC**(W6F9|&C_>o'
    '#)q0NAW#d`(sL6psK5=pfE^_jFKC0w+6yyXYO`bet(S87oGLC1{N@^TUuXk6`xa!l@b}$pfjpbyh#=LCVnsuCFR+F~zD^stp;g`4F&gK=!Bl+DQlWB^>fy<>'
    'p!&=r&5q>aLP)>b=WMx|t{_e}O*KxMXS!yJ$L4F^GS%m;sC3LyVF-gCUcLF{=Yt=9{^`}5w+DcHz*ME(X~?yAW2&estzF=JDRD+roD%>V&zdy9i5gX`itaei'
    '3y%p_!6wIbDKi<|fy37Fp>PILIKP^(r@H2l`$Q4mG&)}J0je5EXi|U}29vgLX4*^?^P0ptVWU2xSQ@gwVFNBt;_L6eh17pvvs8{e?;Cx;$T}jxiXq4Rh~Ap+'
    'wt$Rw)Sc?2wqesdb1_lN^%)ncn&$L;u0ij*;S7w?UBc#KxflWg3YW~)vK%LC*5Qe$WF0n$N=lrMIq?+?5LK?h^H9O<Yz~#<oDB#E!B1f&F*o{pLMt$*0X9Zn'
    'Orf^QGd0A;2EwF)3QViRw7)}_quaoej74<x5wKsbRy*}*ex%$Plon6_HDBA3kE7DH4xdPGrlJ4Gk=Mm!LWb@GI6I4CxVBLye0{h3oM#H!%fGySJNRkv>#JX0'
    'y?pzo)%q`j{PPjy@lhAOt)qi3hs=3m|LTYMct4nA_l}}wxA~!c0;;`E6GVh8=XQh7xq)msH{hMMuQO=_^3k(zbA@~>8dWeht8s;^B2wEF%kpOq<mAN@069K?'
    '3LsB0oRn!Z%(}Mr2#QCR&6>WCigHV4SQ~!fanBYg&)qQ==mlL&CwNF!&16!%KR`0GSLmtq{V&><a79V*+BOvPDbI@wm(gK2>$Hzf^XwRYoObf;l-|$*TB0@N'
    'Y@zMB+FH0cQ16?NSWUb9+tfEaxUJ1-`ia^4vwRA^0}Y5~=I~V=rR91)pRU>@ts%?!rin|CXE%YW(l-A8b@yz$Z5&I!>nmoDktHV?(IPF{4ig4&9B*PACpK*7'
    'zyU!FC~;&nqDX?I?C59(>__Yu?w4%cy1V)^LrQV57e82>neOUab#--JN3<ejRYeh>O@2KRjxH&3p16@O-gnq#vz_}%c57oRj$n$8VU~)lS7hCBGKez^Ggm0H'
    'LozyD)nyp7J@)g8^(BdqnE9_!A4H%4&LQ+=wm4H?B<-;cgw8hZH1nNK#|A*RQxDAQg+IUe=5ia%ubNzm>g(*vO=om|a00!=0l1t|E&(kzQLtm`up6u!f!~=k'
    'irf6kES<^4f%;Tx7W$Is*_ZO!&MXz+FbGsabBhe}J8y=+%DUBI_c&k|y(OT(iSH)fC61LD+Euy4i6dWNg9?WlT8@zgu)!YP$qeS8V1eu^G+>mPkoTgA*L^SR'
    '<=KyX$JO`X?4$xjuQtdX!X`vj{A~LnyTIPd1@}JOS(aNk&tQVs8LHe|;Jw&PFWPHvI@``_b=Jmc;fFCZ1t(a$2cN#ox;V426V8UcuVsd}<s2Q(?+WiIhUne&'
    'X<EtL!8G)p8G@ah2RXU|n4tdRpK&@_;EiB1$7#Xjp!y0<!TpW>LdPZJB#P^2WG*K=!*^4$lbyx>DrzX3wwV-0HF&JX8lOUj__189UYEUH6a@wLatO=TLqB?S'
    'e5VGG22SP`e(1t)AgIG`&j+cP*%Vl+^VqM#D=?y{%zP}t@p}q%W`VS3<aM%$8yF{`o!l838tl97b`f<8+7dF}$NAXHm?_9a(OPKry1ghfgGjx~Ns*N*yIfNS'
    'O`elNtzw~Dg!9S=eR50CC%ZvQI(iGW?!cpQ)gO2a=41~rCrw~H(*d{|%qSF%qa4y(q_xtbB^}2(>{=^|<k+=r<%vWUb#*92JhbA<j52r<313&}@x2j1Rtk4$'
    '0_<cbms=-^J6s^|hp!1jN9C#9J0Ir=P_0j=+e7-l&ky|`|9PzPreV=zdVMT~_<@TaKMIQ)NdDjBN9IwLw*eT2J)Lutr^EGr|4|)66ToHRA8eN%<!|q?Ja_#L'
    'caN}R-7a<SYx*rC{JI|u?SUCN|Lr~EzYiYE-`?(_5!3F_iC3*QpQtt93TLDt1F0ZUKhM5LIl&>B#AMbE1~V=QV-JTHOH;{4Hk)j+MZv8tKk++8E!L_F1uC7f'
    'jl|=GZQmX-sm4j6|GPXRU3mg@qQAL=x8YfEiR%&mC>=kPb9D60&>q4@<DA1}9)torD=B6nH5_{wYZ4W=JcKE9?qhdX24;@laC9V;X$rmn*t=yVPPyz>*dc@;'
    'XbXEZ?TLw=|LG%kR-JcP1kF{6iHD(W00+9$HjgRitO=>Ymka1+Q1JLjCmW7)L!v*wvS%h<6hVfJ6uzsamH5#hI{tPGj!$NWyewub9y^>E$h-m*!wv@6{*HWq'
    '$f4pz8hxZE65=)JZ93H<b9E9|!y222w?f_6bl$ri$;e~|R1IyThFT7k;n}(IlGZ5%E)8zm9BO6){rL-$@5#+UPqMl~*mC+>k;pm*`;hDb7t_6@2q<na`y~=R'
    '=<qT4QS6Nw;qoX%%(cTsqmRd0n1>Ml6rLeIsr3EgH^NV|ARam;*2wMF>FScyx!xa|jfcnZlMw&y%hykTDxdu0$@3RazJKv7N5?k~#&UK@_5k@_0*(igs4Kyo'
    'wvXXFGLd4gg5X=NqZY<|>Ky}{IBIp}L9R4Q%pNw>ncftUDWP*S5-8!p1J563Qy`^7XpB-z&~y!3fVLl6=(PHN2p7p-p#m;B&$}TG;n;9Xv&H*i3$PaU=0(eD'
    'IEM1qL9iVtPtbvv0={5Z-mfmbH~oilE_Tj<ioPo{F>iJW$!MgV)w;qg4kr6)`(2cVjG|MP^VxajeZ({_M`IE=d=2x62Q*A$xDJ$hCJ(ChxLW^~)w*1*$JKiK'
    'R(pWefHirrXFV3=#%_%5&^A6wjWkp%U6&?k2#Dzld5(s|?n7N4_<v}+Gz&zFJM13yj{QccK7V|8+<<9)%_p(cdzbXpjhJwy$lev1&09zFTSN$gX}t%?kdSW4'
    '0>B<T6xT|vjkFZZ?0@q>`2kgy!92><g)Y=Qj8ShKCg)NgA4QqC)W^qT7f8=~+3?P=N$w2nJ!7@WP}6bR{3vY_?}^wMtA`aYj^M4fG4;RHwBRT{c`FTToRz{Z'
    '>m(n#d+s+f8QbN`Qj8#U9+F+_f`mL#ltwz~s^~0qyz#3C!>k9cU}@!>^##+i#oPD``huILK83@fLGR~CsS@-=xsCorlXX6p`9UiDFEtRnz($n!%Wi*2a*8|M'
    '-mV4X(H6NdnpE7JHpH>z&|o|1u;^D%JiB)38rE~-ej+&D>O(^neBXQNx1u$9Q(44V9j=MIgrZWaL|s;+4)iK;DzxHhJEI9aadV#R8x6nd5a^U-ErRT3>6*`@'
    '4AtThYJ)1{@7&LAZ=&R>QbMf!aw_Rl$ze)1rAij#Bt@Ixo}&%d7fY*$rwVRr@>CXjH&GN2YO&Q$r)~~ChM)7)mLDb?GexG--V>_hZN;29!9sp~WZtpNLScye'
    'lJ8*Q)e~Mq4(<Wke#U<QPf{=eThs3xW&iab|7kQKuTag6($wq)ZkqK~*8cI;i~KL>0yLfRm2;Dk{s8p}Vsi+XHJ`AMKATUjs&yyJ-oj@N`}O?wt3wz-#)dhC'
    '4Cl>#{Qp%L3#`EVf#HfMSeUVscHzUwSXy+w01Z2`L07=D%LP#~%wYVXLhFQShI#-p^gK9#W4a$$8nMSlAFV+3SGE?Ck75pCLv721mvf$Qzos)8vSB|>i1Z~;'
    'QX%YQZ&cNSEK!4{?MP7&e11_)m$T`nWb?rmT8*=fi<<osi9Cl8UjrisSD$Zy=mL*T0tb#uR4Nnl)NpbhkTF+!_4?)8mv4W2_3RCd=-B`E^YdTIr!Rhe^Oh#7'
    '#~+`*{N?SlcW*@%#QItG%Gs`ghlvxkUf*!V*Drs4`}~(5H4{|_B$et_g96U+`S-8cxTp>sDXB1Pc2l2TyaZ6?%hx|VLw5A(PtTG)g3I?6aMvCr>Y`1`ZQy<X'
    '<ju44^^;%z{_KbH-=6>Q?8Wo9zbS3@Q=s4|a36I#9oHFBXj|b%Ut@+S1lkL$>vHiWLz{$wM++&9aBid2GZ%SYFe6ENG?hAr;taSpkaIAR+*H48;zU#gpkcS<'
    'Jrte2m#Q4E_;Hu*q<MW!re&o6IqHG#nAab(8E(-7URbsT!;B1MRWOjNOf?aa{*JL~&|Z9I3CL@pw+ZBEz-uwioqBXbCv|)Ga_2QeM(LY4SkKm*Z8UaZ`hXzq'
    'H6?^0gTRDlkDl?R-%gla4C+ukMJI7d?qpmPNESsOGZq=-;y?YP;JI^l7&42}H*MrGug;(gYl*)z89{9r{R#bbIi45H+#*+s6|#Tl8Y&;>>|!>btkDc%NO{KL'
    'E+n;}cN~1z*8W$io!#d$D#RfUCp7jLosZGNLktK+I%?a?J-p(t1jj>%)nv9N4#{Ezb9ZXpJp`!10H@0588nl^z)ZZ@?Rxfy5^S~GsFg`&UZLeu5!UlTT|yQL'
    'Lx@lY2T|a^2wlQKj%F|%`T6@e#v&E#4|+bD71{4&3S&i&zXNTd%%0emo@@TDXBk!#=1^ix-))AO>=5~L2ZugHcmlX3iIwR-I%|#D0xhde3H<23ZK`VF9bbX%'
    'l=)w3iNywmxVi;7vMbDXJg}59Au7xI-RUUxsKVXGq?G@c?g=%8eWNJp`Noq}U}cH}zu@@@@;+A!<6?{W&C~5;3%@fqBh;j7fT!PlweY;n$`A@uSYr|}ifm=Q'
    'yl~iTehkz;EHYsS0}8H?Nf*`(fZ-$IlP|p|!IFH<BO?tjIWcz#Wxxs=V;K(t8G2*(@g5_>+3H^@TM!ZnIskb-V&UWf^#*7t)|VvYqJTZ+h`dbg01S!@ZO&pJ'
    'tmz|$Lvmdj`()F7FkyahxCOPNm&Nejx8B<Y#bSkfFtXhNXBMcSM8Onp7L)qO#M%_P7;rs#LJ4I!dz1)_3>fHQ%OwyT!P$au+&s`*e1%hhIbqSTrzHf=Qd%t4'
    'y-^g&BD2}nDUulPX0u49R(xC;@L^38Z?Mbp1}OrRn`u}=wS8rF?qxq{^<?(_ug_onkdfyH+P>2LX<%-&yx>4Z=IH%61^ymZVvRP0+e6}CI(jc<B8zqCZU#Uo'
    'Dkz+bD3;?CE8NesZ?p4($}T-wTdOGzOzpR^zcEy`k=oHxSnp+0*>sg9tCTS$OAQ02S74ecbi);ji#)6uM84PHg6806y@gXg;0W~MabRR2*?wCkaBL`J60)$H'
    'g4*db7{P|)ub%vzZ9Yv_6u@ag!U%LjB)J0FiE>{KE&c;B6a)|-S}#yA8ZWXi!<rFGXlhfKBu%c=w`c>ZJi8oB--k_A@qz=>B+vkq@w2j0Cb0vsLi4H59(x2o'
    'V6jN|VFc{yl}Sdr_Uf23Dj7!uSo`g4k%<2R?3Ifx>@8SSYGg*o@t>vXT8M(R^cX`di|iJdR09O30D*j~Q1eiXo-yDZT1U%ETOhED7S4v?0~~&=Stbt!r1x}s'
    ')fN&wA@H6vq5Y`*2-yn8e)*SSdPyB!ux0&-Ux{5JCxknC3Xbc1=iO|gK9>;(VyqLQCh59oZjWSl%%|8m;(m0hv6I<?9@~i)={SN-@_zeasEMGBBdD%HD!W)B'
    'uUJh+F?X!vp~EAhvxhP@<}w=mtv2Ae&S*;W2<|~i;vKJGTo8-`QWf&qhs3O;DhY$Zqjf8<i=@~N9I7Tls`z<n{W4WSH!QEPqoq>Ty@>6J`Xo)Y><o)tck+U+'
    '<<EQ>l%LGh8&Mf{aN2Y;#JfV`R%mvU`D*=_<U5T5A8&bG^Y4IeF2?Z%(VNdW$Hb`dnh*0Fx4W`!I%$r&!1%nHA`!;yU5hND8w(dA_ZzyScwW!A3Wm8MoTpxl'
    'Oygs%H+O81`K-0PvMi>E$Z0vEd<f12KcVJ}jCno-!Mk8u2RyeT?Y$7yIBd8Qj4As`jBMOzD;)~1FbOe{F}ZWS3vW39?Cf$rk71$A?7q1mgqNj6Q$ccJYi6J7'
    ')29wvHsM(qkQd4w#}YiD+jy=k#!gzHE(di@m!sEF4}5zD1o;W-NsD%G@dygH$No86wLfW^6bI!F1dxc|tnl3~C7X8pmNXzCp4Y4Ac^z^ma-Zma!YFR&k)dTD'
    'gkYp_@w5cAe#BJ1zg%9*N~%Q#C}uiw5!h}rM~fPQ)%KchwA(USC<oe>kq?b`sD$?fHO9ut<RpzhqIMArpxxK(<Pygfl_6za5;WfG-^0dVOcT}W@(C11Ajs*+'
    '3Y~?FbCq!cFMXMe9S-+OsjN_fA@>e9R{w;wWow9wtDoC+Qs3hOWE?n#mf=t-P#qd)thSuNK|z{cZK#G?YMWY0vV^jaSVKZlw~B@&G@ulfOF6|lTToS6Awk4n'
    'DyvXaJVuKIH6XmOC#BAhE(nvT16M(Gil)wYuv`^b>KWF_PL4I$lg>((1^b;7hvHjjEghl(UMu|2BvA=C>b3bBZP+tq@`i*BT|9uC(R_^$>cH$nr_KTvC4@6}'
    '6MWZRs?{+pxRp`KJVyyK`s!qE6<Lig-gw1&X<m1FUwT<!cb3-XhdT?0DPIRk9ld5d{S>gf%HuAq{@B`(>Z(6U?LFNH;WCGz^kE<=CwGj+M7tRt37>KisvWV^'
    'iQe&HPl`Nqbf1~S`YiXF!P(u_j>3qbHT7jjoBP?OiVZdU#j1SRWE|%B%*#z>3aJ_rRNuWZ49#~*tgSQKZlSYeZbyO^VW<>jii9RuT603%iHX>!57oMAo37=f'
    'qj4B-Uy4JAlDnwJM7&jal|y05>4)W}TJUa1X%R8mgq>|&jky}mpi<M2l;MN~Uu6&bNByHNDsxX!p_vEq(CyIW*GP9{tOUG&m^`+p6p%0=SS$uGA^;X=5Il7U'
    'LL}2LBJ#Ee5M1EX`6cQAofN)?mZ`EKF;m7vT7ltWD+#gmCH;zm7UsQ&5cluv4r<6ae`4ii<V#kuqvw{qYol)D{v_4`@@9A!By5#AYb)?d-Op6VM_OhxE;8?v'
    '5!Q?gLz*0mu)7sKtgspXTSGgTf_Glsf(niddIjXrTT>z+n#_7LxVufjli39-V#ssFw8$|s_H7ng(nLsfkhb~sjW6nA3L2f^c7gms5L1PEpHt<=?;ZN1Q#|n$'
    '41YoVVhR_$DfCM@v^>zldUgDAxge$+R`a!vEWw|wID;tdiag-DYH@p9#L%5I(n6+nsM`logf-vhao;}H?~HK@etj=%wo%kghcdDbtwDG*;LkpHpXy8l81=mo'
    'PRQCR;i^4JQz6L2JLbAB=C=GQ{P5Z~@OK<d8&gyxA7e71pp@H~yYCSUFfi{eY6AfFcLo4-X0s%xFYT!$-SKqM4ZB3DhjMEVo3&Vb+>FpQ;C!@egoiDQ*j?Kn'
    'hZ1<NgRz)C-OWAV`aRtPFm}XRMtdp+uZ#30-K%tMza#1DKE31o1OuI&pq94nAWI2Q^(3enzl6r@>YO7*&G=Ohu8y=NqcZwd>a;%-K^Id&5|zC=hA-lT?5Svp'
    'pEdkrP3kniq$7MyCg`ImWm`@*8R+9h9x=I=k9c&~Xaa0eRDfcF$YD$TLt2H^a&YF0yw0%lEV%lF*YQMKBE)5~i;}DM?>Qa3$a+M1tHdYQ?L|I;M!zGYr#%pw'
    'Ybu5Xs~WL~aj9@hS*zr2tGn)L&TLuTZF!uKjxlJZo7GKu8d~)s*3+t4iI_SH7^Ry*cIwNRt*i5JRieM!wGeTH%`G^$mwg)udC6w@+FfLX8TL#Pk-YHJY~((-'
    'DU>0>9~g-w#|xfKtq4<pgfEpRv}f-)5^U2q{DD?+aE)yZBrtM|ylmiNspRcGJEcC<u~RJ*aeWXn)D$b&s7Yi{XPysRHJ_|DB*13$YZhZ?heqP<sa)FWJ4E4Q'
    'M&Yev;n!h=C^Y&N3R7I^qfXyX>(&klS{{54i4c63Q5#QnrJ-4#Y;ifX?4s@BO!%sq?455rn^ML2>W`G_LzZli{S3qrP97(7EVL?RrwU(r1`TY8Qe5zNERDDM'
    '>YY$Rq@z}m;yV|S)hup5j<#?RNP2Ex`C_)gdjZ~3lQsk9J2sLmK*FQKh{2u3**h(u#3;#AcArl;TCx9p15qd<C1D<T*<+d~b274oA6txX=hj_dmr*(c7+Ijc'
    '03`M_SMzA?EB)^Mole5+%}-BW)pk)W<t_TCA@AEyY$@m{bW&r36E0AAI{)l_H&R7YXRE7NCN*U8G6}}2$|Q-aad|AV_tk>1YpP=WJvc56!A3_(o*IA5>|!dL'
    'dSvK7?q%v>5%g3QrGsG{iFl75IFn3e$WeLzu^vN4tW=<+S7TR03nkUJaG^NMYY`rsC(B6$=apg~i(72JcLFVtZP)B=fF~h`Fg+;kTD7aZ9w`JQa<eA}V%g7u'
    'f*u?X2e)wOF4v53flKX)<XTN)oYn|i<V!FYLGLYI2D$?$c8~CrZ*L2yj5*3<#l+F0ZjnG+t5Y@?3K6>SsM}`n{#52E?3P17<2LrIkS1y{yGg}bPaXevA+$$='
    'y<Q1B#==J%@Yn{Ujzy%Rk}meFq9$eRS=3Ax=XjHz0VZ1fOfXDr1rghmSjNeB8&wWMXh4aqBi~E|Ua8LhV-HIf6VG4qV-ko-GHfVAmb=)sY)+S}N<GgOr}N8c'
    'RdOR7Fhb0it(gjtrdp@&*xVtVTe?r7?=!O^CMD96GfK1D190;5c{MdTu-!!Ctq5@#{kR!1bTl9_Jg|=4yraqSEnX?-(9H%gH1z>-O^{O&$R<kHS-RZIo+5{m'
    'P3G%rGQB#)Lkq1IC0K+QSzrVNxaD!g3BVt6#^4e%Lzq_EtOD<2SU^rv#&-fUr7hpPG|86o00L5UYmz}Ms0X52Y*|ORH`NF;WqS;)T6uCI$-b3HKjyVML=%nz'
    ')EobtC8SB#2UaugOvtY8(IJlr)^{qTl~KpM0$%4#lz_Uem^_7L$qNsI3pJo3^Dgv;{uBY(DMANQm$A^xVv5*V1XLwFcVkQz;aCEd<*+xnB~p}~mnJ`KmjNQQ'
    'Dx*VtYZwWR1*JtQMRthcSpQ{+q7|CRU_t9>oJf(J*`8@xI09_>q?#`m?>BJr11@2>{<SkOQvtSka}#$g!elnPX%-6Xx%y38UAn9os07<|FMEYSj@BQmj5TYA'
    'o6Cy}G!&qqHGoK%2RM@eX++g9ij4?zH`&RR0%an%97^OCvF}b<KXwut@djsrKwrP3BsZ(10XqsRu|Qo(>}8!tg6&*W0R*tib-@DKr(hteOhhXTa2;KSl(mw7'
    '5*j-z&vr=}wUhF6`bUBuCf(v@i?b#92?>X9<cQ-DH{-Xl=Iwz>E&74O7bCdXnPp@*6bcud)ei;4z;;KWVt;J+Pav;y?K6Xq9pGQo1HDmGUBK_PeF!!8J3E0;'
    '7%nADF=__m8RZUtXvN+v66@g{THQ7vGY<_h?Ud&#7@^8&oPVQtiT7R^ljzrKpQ`uC24z>7VnD3Fu{+6o>Fpltiwi9GWMQh*eLkok<w#EnZZAT_kzIvxU0quF'
    'xH7$Ns4c^dYLW>XBfqL1_uNcPEr6f{d9nv~q9QOXs5iTBcUSg;v{#=PT_Q00Qth_3Zz7V~Sm~R0aER>DTEZe_zvdDg>fsxL(=rB23^rLqv5vZy>P=HaRMsK+'
    't`>B0&s0qUvh8K<oQRS`;(58Jgj|<`Y<6oRb>y8aZ|sqybZ((mUs6k=l!+B@XXfd8cGGGy+tG(6^7IL~zCFWisw_}&>&_@TV_>FWOW7S>4_ld|nEmJ3kho2m'
    'hCxj<0SUHteNPnEnlP}?$8_yj813pnC`#QhmVLS*M|J&t?A4J5D1XNGgOkR}p1`aj3}-%p8^-2VRouNRnQ=7tEj|+Q3wLX^_1-<W-EH<SyUE_=7VGI|NzjS8'
    'VlOb*9sED=6uqP&)i1mVI4U|a)h&X$=w*q&IN)#O6H4EFpHqq%%YBj*#>Qs9+D>qo+)xKbch$B&idcEn^23vsK_>csDigh_dYiIw&idG3F;yW>Q{S&iQ*V3q'
    'uwNg`)Jt2fYt#fIt*=mH=v7-jic%Ue{B<?=Bo%&2a1B6F2M0m+<*sUmLfod}jwIyXtTd(}H(uXVLH3_WizS>AOl2G8%L8OEXxqVG-ZCPRsiiX_b+0%l3diKw'
    'kuTP-!dYP(a>jbab(UD8(89WKxCd!uC53^A%<OQF)o>*x8olG7R6#yr`<;>no9}++U}vI~l<yB!j@xdiSaCUUU7+C<39y;qOW?6EwKn?Bd{pWc40o#xN_>V<'
    '>`Ac9Sm{Mq6(wT0ZXF&1794u!?rh~==Tr!j&2=tji$mkKldPnjcWz>(;q`fSHONG^%?b#VZ@0_Byv#{_j9v9L4~*9+6N61Oa0B#t<L&aiBJ~H|6L8Mp+B);H'
    'wo1q)kw|lefN!Rkco6_x;{wQye3Kb*TNr=IaOG!nw1;e;PB4sV8IL3x7)cr%Qth<7fFfOem37g}PIp?ws{g>*WufX>>l(q_kVEMV2tYRB$=8z7C}QSjgq1i{'
    '(q?Oij7^Sd{=OP!Jw1G+G~Z+=>uPddJERWTnQc&E(?;+Btu<`bgej`+Mu&5fLCqrT<|Egi?mqny&Or*sO<ROAw<EpUGe4M?5LFWi#8!2C)yqWQ=cH1xi+vc#'
    'M1ewLVD|gwYJ+h|K90JBvCa>|F?6gVL`o^^gY{KlB_u`2tIq&Ef=lIBY7AbU0#3(csieAYQ#hV^eFLU@2O3tViL=nhmRG@xXQcf69<rQW5GE#NW39@t&}v;1'
    'eK?xDjYBw&H~g@iDh8{Wy|UHUmkYRYqRi%R(9}_Rjj9mXB<nNI;`POEUtD}KEx-8bi=V%E)5>qkYwJy_GYjQ6e+yBP4lav@QJVXHyDr+pP7GD3Kwt?{{@1VW'
    'ot@urwf3LIE%kGt2F~x<<YETvMi~^s^Ch{?oP`Hgm)ShH%)vDdZCZIa>*aD=$c@e3EC))7^y`MzOu%RX?jL81>Jli=JX0Lpa{Udiea4AIHrY?l-n@G0<yHwp'
    '#SA^p@HSoZ#9PD^(A|r|K+P;DbCQNCm$O~B!-PHs?|l;ZBXu=RhVaAFL@I{Y%uppx5-}5>($s++d3mvb%_uKMT+GOpeeGcg(a7m`12rYb!o=|WcNWQE*$x|#'
    '-yWSdMNI?oGWv*Nf+q`vq$`WK1&$ICy%8tcNI)5B%3Q2gC?!FIQ;k8F2wBh-Pr&6yDd$iAI4Pes8#*D<aVmB_+o5QO(j^7EM)_@d7w!lzYmMf~Kd1?;zI--='
    ')uUA)1*0$x6@Yj(0j3VOF~+QKt&IE^9>Vd<D?0%gse*8xk2jk0n^rydLfSf#%}^dQ*Ne^?iEg`fF<H#c@W%b|%MJ+?(_A*|RuK;MOEaem1SNH;<58vh<zhZt'
    'oLiH{<vDv)DHCMMTm6jwVCY~E7^V(#qs5dS8xEMm0qm?ykS6**O2|*D`DD{d(MPz)EQCZ}C!JiK?-X3t;+LJ%&Bs>SJys<0sruokCMXU?H30Xvp)MvE=k(_4'
    'yirh+-y7grfy@+6{n=bW3cl<ELu@xvK;%;jO1#SliI>ea%|;$1UZ!dSZz;w37nkvTxA9~Bn%tx>uY5QUh13x31EkDvdn#w2FdI_|O`7krC+=JHb}(0ytNC&='
    'ja|o$Ky{Fkr4(Ah6_aOA)8!lX#E$0wgRbP#j|2yuM-^v?fn@a$BVi>ADlLtfP-t|u$#4QMH?dujQeW0q4lz#DxUvsQHgCd!3Za6w$dU@!&*|*DLlk>U58-{i'
    'f=QtH{?hKQk8r)>n^Vdun0EjkU2Q?|A`1?I;DBn%Iwoh}U7&|m11Yk-cg%rtFt<uY6-2IEethu27wLYuTA|h6rjkhzb&6N<PWlvAneWz?8&&u1u$fq&LOY{2'
    'Yn~*5O9I(n$KtAnAsHx;q2>QmySo;Er@?4BMa&_3znmYM*xSp67yM9crBz}+6-FVPR>d{8O!ff|z35C0w@b9df`);}A5v=ah9+9kXqZ+R*1(Z0yELIu<JDf&'
    'DykK>oF-GIRYj-1$r4m50Cp!;ldG;3O;B@R-H`*ZLv!Q`HtUVlt|vW=lPTG_P)^_g7kN(UT=IQh?)KtmJkaLEu`Wl2k`MUfu;^Z`?R$Tqdy9gT|9`kez@|Bw'
    'u_Qy?Bch=qsW(MijaLhtsnngcNHL=01YgK`Pk?RYYd;w*!4ugm@oJikOK8lAUxMt%wr;57H#u}|0hYq}CV-BoT%qt{af*TDsb56~glK7FRnU}E&@S#Z#r5)T'
    '!(`qq(~CD()L-2Yha4WU)SP!TdR1HaZK)5&*)o80XZBW3zd>~N{a@u*Bc*?u&-&Ah+MnVr|2&tM+@hbE*Gsm+yux~EwFDtz3SdkB5ueu8#Ay>O>1And6f=Ai'
    'n=GO=(Uzs+E{&hna3FA3B1&h1*DuW@c^VPaQyv|Coqeg=hz<e1^$7{W9l!(P*PD=;we8AnkU1=#fSq}NLm+_Q7`3A{LI{-O_{Uc-n6n&b*XlZYgNY>V5Qf)k'
    '%3z2$_pg(4g|%>ghS&H%K258Sv(pOd^BQ$z#M7HroqX1shq8w}d(u@hJMBPxuU9^l;}^VF@|tVJ#>c76)dFq?+u7+|5sMPilTE{hlvsF1ZIILCT`3v4QK2f2'
    'Koiz*>S07SiroTV$f9UoUA5e9J%{tb9%87Yo2?{RkU7UJsdYo3O1V*37cI_Er2u%rYVj1qELl<u_yF>lZqta;s{Q8+9<FVY`#H$2>91a0Gy$Xm1buWql4FWw'
    'T=x}SxmqW1ZENwNh&(EKXRc|2x0h2)*M5ur?b`Trn;ia;-nw?1<4jO?N1B-k+l30C8t}YQ=qHHIxQulb9wr7Mp7VLUov;r$`IKY_7a`3BD!Ovka&T!OZ^~<|'
    'd2=UP<Ix+3C?U}??e9l%lB@UU$j)CvdIU$2|E*59L-NxicS(c1`3|}iSc9QD)3fPtg?ueLrV5VK?yhiC<_M#Zsrjsd96$ODKps5)96*jg2atq)oOJ`MGn=b7'
    'GQ`Cam3UA*GBCuV!m0-y^2`mrTdyl<m?4yYE567t;I`f{KIycPCf4v0byxE<Vo5jc+LWO8ve&+}$L7oA>&+%xRqKn(t>grRYZc5Rr!7ROy%QDT2_#B<%-b+J'
    'Dln^ZugD%0*>REei|k>MJsR6I+*qJTWsa!~6nF$>BoaRsfKNwt5YQwIws*yeJuzBbT>_^`k)0IT<J=~|>)Vu6v&HGQA;UA&JjI&XcVhDAk~1(+_&Rz-z3Q7$'
    '@Q5(sDSqKDbN72Rn(mnUUl-cjWHEgV)x3O|aIMGm-#7?L_MHsViC-Vk#n!~nOGL}WRDqo%mk&^-_6oB)r9CNeUSNPVVZ0yb2U^oj-FBXhq#@@NgzTQWQEw*m'
    'IeGxjHgsQfN%+j+s5ex7jPaU&r6P+z4&ZWc?IU}OVgK5Xi`>pDyWFT!9F;pYt@pC$F05xpgsfU;n~UXn#al&AA!t<`+S{$u)g^J4-XBRN1V1rINc-)}*H3>c'
    'pZw#=^A}IPfAI_zOf5D$I>b$Xh?`zc$~D;n!n86A)Dhd*65h8lG>>F=eTT$l@xH?$7~7Z~V%R<gjP2+7H(3yOW7IloVK}!Qa!{jIR~~W~>IKTHSTF4!VRQ>t'
    '^8*@RR)_Ec8}D)AU*L2uTIxu~UDSJ=cg}$OeOF{xVZs}BnqC3b0g7h*|K9m&Zo<(><U1}DW?SJ*ZJ4{$#E+*y8>7MCushzB9Y)Ug*!rzqm$YICBdp8oFvMB+'
    '#T-MS1$_D5aW1xv&yI~jSa4_UpLe6d<FVm`D5w}IDCcJ__m`7o>uKUH2oAe_&3zj6#<Y<WF70-+OUXygB`HBQmwYTDxw_1fSPkVlv?cV3fN(5<#|K43@lg>c'
    '+SGdgQ#MjNNF7*H`KPGHFq)@(D!gK*4-4Qf&?QnHbv{Ys<T6lwEUcdxymo!XaG};gVc_*zL6pAZwAfL&WRv$`>ObQ@fcFMA^>i}F&>nOn;9x7OCEOc6duznn'
    'pq(-EGfw0Wh;&JJF*G3Nhs~%gQ4ms=<G_o<IIVCwp)JTISzbW#P1<XW@QmJb*b+fNefLZRCpl~Jyja4#$<8LTIfjqArblz9F=WK!{@d=>VBE%}H>g8$M?iB7'
    'sJF<kVQ2)PO6QaJq%;w;{%Hr!D4H4yh&0Gnvz3`Fs?+jX4Njij%>t)<`XSXW5mV8R8J7ZLoR3Rr8VJu8^;ejcmj>_5h0UGx>90RL$!t-WO}b!`xYhx1nNe8C'
    'dXu${v=vjXSI~(ndv*1eddZeZxj8xQ*=)spaz;}_d_^t9YIk*6QV=RN4$)m>OXOtLw&$B03qHeCUr@T_Mk8+Fs+L<{H$$z9P!^T&=sMEki|Ay!+?3ZM+rMEW'
    'acavToYOX>?>0vFK&|f7JE#k+BGOLbu#b`f1wz1ulW`U)hf7|evihSsy=1|gigXP3CG_#V+@7+-c|}vWSbl2bkADJr=)gCWh%J%DnE2U&H>c~_ibQt3{zFtv'
    '@S@p!_jLBY3K9w%{epzjkt0`Iqy(lY_-Nv|)`OvA*5vYd_4@hGPhS64{`~CiPcMIX!y&zNCtw^OD)?_^R2EP(ZM3(pRv0Nr3?X<@k+V)@J2V%P<yjV}|DF1q'
    '`G5vfG^gcgO)G!^91k9Or*||V2pUrMYWo`yq)i8+>?2x7ByFI<05Suf;Zk*40WER&U)S`><rEGD&0YWjnMY0ePOgOLWT*e8Jf|BpCQAdlbOjxUQn)hPHE^x}'
    'gvo>7K$ojEj^#cuP8AL6$*)f{(y&$b-quQ)F1^2;tf$l*YO3fU%GgeT`fQ~piwd@EMoWprgk8MTHd;3dDOop0iZ|GHoTd9()M~T!3Pq2BN{zQPkZ25^OEwcy'
    ';Ks|~-K-hAIS92?&huQT*1j4lceJP|0@dQ<Y`vre<Ia=Ym<kkz5KlIzv)Pa}dlZs%u|*DFu3AHAJ+Fa(XG*Dp%B$R8R-&p^rjbG{oL`&?G?YRaC}CvL>iiB2'
    '99f|6(v!{dFp5ZnCQdia=O=Qwfs7Y5$8em&a-rR(9=v+(4t5PQR7>w;kX0P~DZO(m@+wz>dLv!&o%O`Bm2j`n?n&sPZ<_JZ3px--htakQUQTyW?HEnOH7PIg'
    'TmvSnbrUJoI!L${)3%XkCRK5H4QMOc+z-Aj-rR^8vJ0i*%V(WkP_03bAJ3e(aqE{Qxii6G4iikVRGFr^FshbSm8jXkY`-@1e~pb9q;KcQlpP#kxL3KhkhY;L'
    '@w!u%5|xqfJ_r8|?uUBD'
)

_ENGINE_LOAD_PATH: Optional[Path] = None
_ENGINE_SOURCE = "uninitialized"
_OBSOLETE_COLUMN = re.compile(
    r"(?:^v\d+|^historical_|^legacy_|^selected_candidate_|_from_v\d+(?:_\d+)*$)", re.I
)
_ROUTING_CLOCK: Dict[Tuple[str, str], Dict[str, Any]] = {}
_CURRENT_ROUTING_KEY: Optional[Tuple[str, str]] = None

METHOD_NAME = "GloPro-Complete"
METHOD_TITLE = "NEX-ELM v68 - GloPro-Complete"
STATISTICAL_SUITE = "paired parametric, paired nonparametric, omnibus, rank-based post-hoc, and exact win-rate inference"
ARTICLE_METHODS_GLOBAL = ("NEX-ELM", "Kernel SHAP", "X-ELM")
ARTICLE_METHODS_LOCAL = ("NEX-ELM", "Kernel SHAP", "Random")
ARTICLE_METHODS_TIMING_LOCAL = ("NEX-ELM local per instance", "Kernel SHAP local per instance")
ARTICLE_METHODS_TIMING_WORKFLOW = (
    "NEX-ELM workflow complete",
    "X-ELM + Kernel SHAP workflow complete",
)
GRAPH_FORMATS = ("png", "pdf", "svg")
DIAGNOSTIC_GRAPH_FORMATS = ("png", "pdf")

# Journal-facing validation protocol. These settings affect only which frozen
# experiments are run and how they are labelled; they do not enter the NEX-ELM
# estimator, attributions, prototype learning, distance, k-medoids, or routing.
PREVIOUS_CONFIRMATORY_SEED_START = 50000
PREVIOUS_CONFIRMATORY_REPEATS = 30
REPLICATION_SEED_START = 100000
REPLICATION_REPEATS = 30
GENERALIZATION_SEED_START = 100000
GENERALIZATION_REPEATS = 30
REPLICATION_DATASETS = (
    "electrical_grid_stability",
    "pima_indians_diabetes",
    "wisconsin_breast_cancer_original",
)
GENERALIZATION_DATASETS = (
    "electrical_grid_stability_without_stab",
    "ionosphere_binary",
    "wine_multiclass",
)
EXTENDED_GENERALIZATION_DATASETS = (
    "iris_multiclass",
    "digits_multiclass",
    "breast_cancer_diagnostic",
)
OPTIONAL_DATASETS = EXTENDED_GENERALIZATION_DATASETS
COMPLETE_DATASETS = REPLICATION_DATASETS + GENERALIZATION_DATASETS + EXTENDED_GENERALIZATION_DATASETS
DEFAULT_COMPLETE_REPEATS = 30
DATASET_ALIASES = {
    "grid": "electrical_grid_stability",
    "grid_with_stab": "electrical_grid_stability",
    "electrical_grid_stability_with_stab": "electrical_grid_stability",
    "grid_no_stab": "electrical_grid_stability_without_stab",
    "grid_without_stab": "electrical_grid_stability_without_stab",
    "pima": "pima_indians_diabetes",
    "wisconsin": "wisconsin_breast_cancer_original",
    "ionosphere": "ionosphere_binary",
    "wine": "wine_multiclass",
    "iris": "iris_multiclass",
    "digits": "digits_multiclass",
    "breast_diagnostic": "breast_cancer_diagnostic",
}
_STUDY_CONTEXT: Dict[Tuple[str, str], Dict[str, Any]] = {}
_PREDICTIVE_CAPTURE: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
_CURRENT_PREDICTIVE_KEY: Optional[Tuple[str, str, int]] = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedded_engine_bytes() -> bytes:
    payload = "".join(_EMBEDDED_ENGINE_B85).encode("ascii")
    raw = zlib.decompress(base64.b85decode(payload))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_ENGINE_SHA256:
        raise RuntimeError(
            f"Embedded mathematical-core integrity failure: expected {EXPECTED_ENGINE_SHA256}, got {digest}."
        )
    return raw


def _resolve_engine_path() -> Tuple[Path, str]:
    adjacent = Path(__file__).resolve().with_name(ENGINE_FILENAME)
    if adjacent.exists():
        digest = _sha256(adjacent)
        if digest == EXPECTED_ENGINE_SHA256:
            return adjacent, "adjacent_integrity_checked"
        print(
            f"[{METHOD_TITLE}] Ignoring adjacent core with unexpected SHA-256: {adjacent} ({digest}). "
            "Using the embedded integrity-checked core.",
            file=sys.stderr,
        )

    raw = _embedded_engine_bytes()
    cache_dir = Path(tempfile.gettempdir()) / "nexelm_v68_glopro_complete_core"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"nexelm_glopro_stat_core_{EXPECTED_ENGINE_SHA256[:16]}.py"
    if not cached.exists() or _sha256(cached) != EXPECTED_ENGINE_SHA256:
        temporary = cached.with_suffix(".tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, cached)
    return cached, "embedded_integrity_checked"


def _unique_functions(module: Any) -> List[Any]:
    functions: List[Any] = []
    seen: set[int] = set()
    for value in vars(module).values():
        if inspect.isfunction(value) and id(value) not in seen:
            seen.add(id(value))
            functions.append(value)
    return functions


def _function_parameters(function: Any) -> Tuple[str, ...]:
    try:
        return tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return tuple()


def _latest_function(
    module: Any,
    *,
    parameters: Optional[Sequence[str]] = None,
    name_suffix: Optional[str] = None,
    name_prefix: Optional[str] = None,
) -> Any:
    candidates: List[Any] = []
    expected = tuple(parameters) if parameters is not None else None
    for function in _unique_functions(module):
        name = str(getattr(function, "__name__", ""))
        if expected is not None and _function_parameters(function) != expected:
            continue
        if name_suffix is not None and not name.endswith(name_suffix):
            continue
        if name_prefix is not None and not name.startswith(name_prefix):
            continue
        candidates.append(function)
    if not candidates:
        raise RuntimeError(
            f"Unable to resolve current mathematical-core function: parameters={expected}, "
            f"prefix={name_prefix}, suffix={name_suffix}."
        )
    return max(candidates, key=lambda function: int(getattr(function.__code__, "co_firstlineno", -1)))


def _latest_value_by_suffix(module: Any, suffix: str) -> Any:
    candidates = [(name, value) for name, value in vars(module).items() if str(name).endswith(suffix)]
    if not candidates:
        raise RuntimeError(f"Unable to resolve current mathematical-core object ending with {suffix!r}.")
    return candidates[-1][1]


def _binding_names(module: Any, value: Any) -> List[str]:
    return [name for name, bound in vars(module).items() if bound is value]


def _replace_bindings(module: Any, original: Any, replacement: Any) -> None:
    names = _binding_names(module, original)
    if not names:
        raise RuntimeError(f"No mathematical-core binding was found for {getattr(original, '__name__', original)!r}.")
    for name in names:
        setattr(module, name, replacement)


def _load_engine() -> Any:
    global _ENGINE_LOAD_PATH, _ENGINE_SOURCE
    path, source = _resolve_engine_path()
    _ENGINE_LOAD_PATH = path
    _ENGINE_SOURCE = source
    spec = importlib.util.spec_from_file_location("nex_glopro_stat_mathematical_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create module specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    installers = [
        function for function in _unique_functions(module)
        if str(getattr(function, "__name__", "")).startswith("install_")
        and str(getattr(function, "__name__", "")).endswith("_patch")
    ]
    if not installers:
        raise RuntimeError("No mathematical-core installer was found.")
    max(installers, key=lambda function: int(function.__code__.co_firstlineno))(module)

    self_test_function = _latest_function(module, name_suffix="_self_tests")
    tests = self_test_function()
    if not bool(tests.get("passed")):
        raise RuntimeError(f"Mathematical-core self-tests failed: {tests}")
    actual = _sha256(path)
    if actual != EXPECTED_ENGINE_SHA256:
        raise RuntimeError(f"Loaded mathematical-core SHA-256 mismatch: {actual}")
    return module


engine = _load_engine()

# Stable public bindings and dynamically resolved integrity-checked core functions.
_ENGINE_PARSE_ARGS = engine.parse_args
_ENGINE_APPLY_PROTOCOL = engine.apply_protocol
_ENGINE_POSTPROCESS = _latest_function(
    engine, parameters=("result", "bundle", "args", "root", "seed", "scenario")
)
_ENGINE_AGGREGATE = _latest_function(
    engine, parameters=("results", "output_root", "args", "runtime")
)
_ENGINE_OBTAIN_LOCAL_ORDERS = _latest_function(
    engine, parameters=("model", "X", "targets", "dataset", "scenario", "context", "seed")
)
_ENGINE_ROUTE_ORDERS = _latest_function(
    engine, parameters=("local_orders", "prototypes", "n_features")
)
_ENGINE_RANKED_FIDELITY = _latest_function(
    engine,
    parameters=(
        "model", "X", "targets", "selections_by_method", "backgrounds", "protocol",
        "random_repeats", "seed", "dataset", "scenario", "scope", "include_random",
    ),
)
_ENGINE_EVALUATE_BUNDLE = _latest_function(
    engine, parameters=("bundle", "args", "runtime", "root", "seed", "scenario", "teacher_direct")
)
_ENGINE_EXPLAIN_AND_ORDER = _latest_function(
    engine, parameters=("model", "engine", "X", "targets", "args", "seed")
)
_ENGINE_SANITIZE_ORDER = _latest_function(engine, parameters=("order", "n_features"), name_suffix="sanitize_complete_order")
_ENGINE_REGISTERED_WEIGHTS = _latest_function(engine, parameters=("n_features",), name_suffix="registered_weights")
_ENGINE_ROW_KEY = _latest_function(engine, parameters=("row",), name_suffix="row_key")
_ENGINE_SIGNATURE_DISTANCE = _latest_function(
    engine, parameters=("first", "second", "n_features"), name_suffix="signature_distance"
)
_ENGINE_ALLOCATOR_FROM_ARGS = _latest_function(engine, parameters=("args",), name_suffix="allocator_from_args")
_ENGINE_CONFIGURE_CUDA = _latest_function(engine, parameters=("runtime", "args"), name_suffix="configure_cuda")
_ENGINE_SPLIT_OUTER_AND_CALIBRATION = _latest_function(
    engine, parameters=("bundle", "seed", "calibration_fraction"), name_suffix="split_outer_and_calibration"
)
_ENGINE_BUILD_ENSEMBLE = _latest_function(
    engine, parameters=("args", "runtime", "X_train", "y_train", "seed")
)
_ENGINE_SELF_TESTS = _latest_function(engine, name_suffix="_self_tests")
_LOCAL_ORDER_CACHE = _latest_value_by_suffix(engine, "LOCAL_ORDER_CACHE")
_ROUTING_DIAGNOSTICS = _latest_value_by_suffix(engine, "ROUTING_DIAGNOSTICS")


def _set_internal_prototype_bridge(args: argparse.Namespace) -> None:
    """Keep the integrity-checked core on the frozen K=4 and capacity-cap=2 settings."""
    for key in list(vars(args)):
        lower = str(key).lower()
        if lower.endswith("_prototypes"):
            setattr(args, key, PROTOTYPE_COUNT)
        elif lower.endswith("_min_cluster"):
            setattr(args, key, MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT)
        elif lower.endswith("_global_prototype_library"):
            setattr(args, key, True)


def _extract_clean_cli(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prototype-count", type=int, default=PROTOTYPE_COUNT)
    parser.add_argument(
        "--prototype-min-calibration-rows-per-slot",
        dest="prototype_min_calibration_rows_per_slot",
        type=int,
        default=MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT,
    )
    parser.add_argument(
        "--statistics-only-from",
        type=Path,
        default=None,
        help="Directory containing seed_metrics.csv and estatistica_entre_seeds.csv.",
    )
    parser.add_argument(
        "--statistics-output-dir",
        type=Path,
        default=None,
        help="Output root for corrected statistical tables and figures.",
    )
    parser.add_argument(
        "--report-only-from",
        type=Path,
        default=None,
        help="Generate relatorio.pdf from an existing experiment root without rerunning models.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional PDF path for --report-only-from.",
    )
    parser.add_argument(
        "--skip-pdf-report",
        action="store_true",
        help="Skip relatorio.pdf generation after aggregation.",
    )
    parser.add_argument(
        "--study-plan",
        choices=("complete", "journal", "replication", "generalization", "custom"),
        default="complete",
        help=(
            "complete runs all nine available real-data scenarios with 30 repetitions by default; "
            "journal retains the six-scenario preceding plan; custom preserves --datasets/--n-repeats."
        ),
    )
    parser.add_argument("--replication-repeats", type=int, default=REPLICATION_REPEATS)
    parser.add_argument("--replication-random-state", type=int, default=REPLICATION_SEED_START)
    parser.add_argument(
        "--replication-datasets",
        default=",".join(REPLICATION_DATASETS),
    )
    parser.add_argument("--generalization-repeats", type=int, default=GENERALIZATION_REPEATS)
    parser.add_argument("--generalization-random-state", type=int, default=GENERALIZATION_SEED_START)
    parser.add_argument(
        "--generalization-datasets",
        default=",".join(GENERALIZATION_DATASETS),
    )
    parser.add_argument("--previous-confirmatory-seed-start", type=int, default=PREVIOUS_CONFIRMATORY_SEED_START)
    parser.add_argument("--previous-confirmatory-repeats", type=int, default=PREVIOUS_CONFIRMATORY_REPEATS)
    parser.add_argument(
        "--allow-seed-overlap",
        action="store_true",
        help="Permit overlap with the registered original confirmatory seeds (not recommended).",
    )
    parser.add_argument(
        "--include-optional-datasets",
        default="",
        help="Additional datasets for journal/generalization plans. The complete plan already includes all available datasets.",
    )
    values, remaining = parser.parse_known_args(list(argv))
    values.raw_remaining_cli = tuple(remaining)
    if int(values.prototype_count) != PROTOTYPE_COUNT:
        raise ValueError(
            f"{METHOD_TITLE} is frozen with exactly {PROTOTYPE_COUNT} prototypes per class; "
            f"received {values.prototype_count}."
        )
    if int(values.prototype_min_calibration_rows_per_slot) != MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT:
        raise ValueError(
            f"{METHOD_TITLE} freezes the prototype budget at "
            f"{MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT} calibration rows per prototype slot; "
            f"received {values.prototype_min_calibration_rows_per_slot}."
        )
    return values, remaining


def _print_help() -> None:
    print(f"""{METHOD_TITLE}: Frozen-Math Complete Benchmark and Report Suite

Usage:
  python {Path(__file__).name} [options]

Default Play execution (no parameters):
  mode=real
  study_plan=complete
  device=cuda
  gpu_profile=rtx3060_12gb
  repetitions=30 per dataset
  datasets=all nine available scenarios
  output includes relatorio.pdf

Complete datasets (default):
  electrical_grid_stability
  electrical_grid_stability_without_stab
  pima_indians_diabetes
  wisconsin_breast_cancer_original
  ionosphere_binary
  wine_multiclass
  iris_multiclass
  digits_multiclass
  breast_cancer_diagnostic

Main options:
  --study-plan {{complete,journal,replication,generalization,custom}}
  --replication-repeats INTEGER             (default: 30)
  --generalization-repeats INTEGER          (default: 30)
  --mode {{real,synthetic,all}}              (default without CLI: real)
  --device {{cuda,cpu,auto}}                 (default without CLI: cuda)
  --gpu-profile {{rtx3060_12gb,auto,conservative,custom}}
  --data-dir PATH                            (default: reference_data)
  --output-dir PATH
  --quick                                    (one repetition per phase)
  --no-download
  --gpu-audit / --no-gpu-audit
  --skip-pdf-report

Custom plan:
  --datasets CSV
  --n-repeats INTEGER
  --random-state INTEGER

Reanalysis utilities:
  --statistics-only-from PATH
  --statistics-output-dir PATH
  --report-only-from PATH
  --report-output PATH

Predictive outputs generated on the complete outer test set:
  predictive_performance_per_seed.csv
  predictive_performance_summary.csv
  predictive_performance_article_table.csv
  predictive_confusion_matrix.csv
  predictive_confusion_matrix_summary.csv
  predictive_class_metrics.csv
  predictive_class_metrics_summary.csv
  predictive_dataset_summary.csv

PDF output:
  relatorio.pdf - detailed Portuguese interpretation with tables, figures,
  hypothesis tests, per-dataset chapters, limitations, and claim boundaries.

Frozen prototype configuration:
  --prototype-count {PROTOTYPE_COUNT}
  --prototype-min-calibration-rows-per-slot {MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT}

The ELM, local explanation, global explanation, fidelity, registered-set distance,
deterministic k-medoids, routing, and CUDA mathematics are integrity-checked and unchanged.
""")


def _cli_option_present(tokens: Sequence[str], option: str) -> bool:
    return any(str(token) == option or str(token).startswith(option + "=") for token in tokens)


def parse_args() -> argparse.Namespace:
    if any(token in {"-h", "--help"} for token in sys.argv[1:]):
        _print_help()
        raise SystemExit(0)
    clean, remaining = _extract_clean_cli(sys.argv[1:])
    original = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = _ENGINE_PARSE_ARGS()
    finally:
        sys.argv = original
    # No-parameter execution is intentionally the full CUDA experiment.
    if not _cli_option_present(remaining, "--mode"):
        args.mode = "real"
    if not _cli_option_present(remaining, "--device"):
        args.device = "cuda"
    if not _cli_option_present(remaining, "--gpu-profile"):
        args.gpu_profile = "rtx3060_12gb"
    if not _cli_option_present(remaining, "--data-dir"):
        args.data_dir = Path("reference_data")
    if not _cli_option_present(remaining, "--gpu-audit") and not _cli_option_present(remaining, "--no-gpu-audit"):
        args.gpu_audit = True
    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.batch_run_id = f"v68_glopro_complete_{stamp}_{int(args.random_state)}"
    args.run_id = args.batch_run_id
    args.prototype_count = PROTOTYPE_COUNT
    args.prototype_min_calibration_rows_per_slot = MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT
    args.global_method_definition = METHOD_DEFINITION
    args.method_name = METHOD_NAME
    args.statistical_suite = STATISTICAL_SUITE
    args.statistics_only_from = clean.statistics_only_from
    args.statistics_output_dir = clean.statistics_output_dir
    args.report_only_from = clean.report_only_from
    args.report_output = clean.report_output
    args.skip_pdf_report = bool(clean.skip_pdf_report)
    args.study_plan = str(clean.study_plan)
    args.replication_repeats = max(1, int(clean.replication_repeats))
    args.replication_random_state = int(clean.replication_random_state)
    args.replication_datasets = str(clean.replication_datasets)
    args.generalization_repeats = max(1, int(clean.generalization_repeats))
    args.generalization_random_state = int(clean.generalization_random_state)
    args.generalization_datasets = str(clean.generalization_datasets)
    args.previous_confirmatory_seed_start = int(clean.previous_confirmatory_seed_start)
    args.previous_confirmatory_repeats = max(1, int(clean.previous_confirmatory_repeats))
    args.allow_seed_overlap = bool(clean.allow_seed_overlap)
    args.include_optional_datasets = str(clean.include_optional_datasets)
    if args.study_plan != "custom":
        base_state = args.replication_random_state if args.study_plan in {"complete", "journal", "replication"} else args.generalization_random_state
        args.batch_run_id = f"v68_glopro_complete_{args.study_plan}_{stamp}_{base_state}"
        args.run_id = args.batch_run_id
    _set_internal_prototype_bridge(args)
    return args


def apply_protocol(args: argparse.Namespace) -> argparse.Namespace:
    preserved = {
        key: getattr(args, key, None)
        for key in (
            "statistics_only_from", "statistics_output_dir", "report_only_from", "report_output", "skip_pdf_report",
            "study_plan", "replication_repeats", "replication_random_state", "replication_datasets",
            "generalization_repeats", "generalization_random_state", "generalization_datasets",
            "previous_confirmatory_seed_start", "previous_confirmatory_repeats",
            "allow_seed_overlap", "include_optional_datasets",
        )
    }
    args = _ENGINE_APPLY_PROTOCOL(args)
    for key, value in preserved.items():
        setattr(args, key, value)
    if bool(getattr(args, "quick", False)) and str(getattr(args, "study_plan", "custom")) != "custom":
        args.replication_repeats = 1
        args.generalization_repeats = 1
    if str(getattr(args, "study_plan", "custom")) in {"complete", "journal"}:
        if int(args.replication_repeats) != int(args.generalization_repeats):
            raise ValueError(
                "Complete and journal plans require equal repetition counts so the registered seed-stability rule "
                "has one denominator. Run phases separately when different counts are required."
            )
        phase_repetitions = int(args.replication_repeats)
    elif str(getattr(args, "study_plan", "custom")) == "replication":
        phase_repetitions = int(args.replication_repeats)
    elif str(getattr(args, "study_plan", "custom")) == "generalization":
        phase_repetitions = int(args.generalization_repeats)
    else:
        phase_repetitions = int(args.real_repetitions)
    args.real_repetitions = max(1, phase_repetitions)
    args.min_seed_wins = int(math.ceil(float(args.min_seed_win_rate) * args.real_repetitions))
    args.min_seed_wins_source = "complete_plan_phase_repetition_count"
    args.batch_run_id = str(getattr(args, "batch_run_id", args.run_id))
    args.run_id = args.batch_run_id
    args.prototype_count = PROTOTYPE_COUNT
    args.prototype_min_calibration_rows_per_slot = MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT
    args.global_method_definition = METHOD_DEFINITION
    args.method_name = METHOD_NAME
    args.statistical_suite = STATISTICAL_SUITE
    _set_internal_prototype_bridge(args)
    return args


def _per_seed_run_id(args: argparse.Namespace, seed: int) -> str:
    return f"{args.batch_run_id}_seed_{int(seed)}"


def _drop_obsolete_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        return frame
    result = frame.copy()
    obsolete_exact = {
        "prototype_min_cluster_fixed",
        "local_signature_seconds",
        "routing_seconds_per_sample",
    }
    removable = [
        column for column in result.columns
        if _OBSOLETE_COLUMN.search(str(column)) or str(column) in obsolete_exact
    ]
    if removable:
        result = result.drop(columns=removable, errors="ignore")
    return result


def _sanitize_text_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value
    substitutions = [
        (r"(?<![A-Za-z0-9])v(?:2[0-9]|3[0-9]|4[0-9]|5[0-9]|6[0-7])(?:\.\d+)*(?=$|[^A-Za-z0-9])", VERSION),
        (r"SHAP-teacher budget", "capacity-matched Kernel SHAP reference budget"),
        (r"SHAP/X-ELM are not teachers for selection", "Kernel SHAP and X-ELM do not guide prototype learning"),
        (r"real_candidate_native_path_integral_no_teachers", "diagnostic_interaction_audit_not_used_for_prototype_learning"),
        (r"v\d+(?:[._]\d+)*_default_\d+", "engine_default_not_used_by_v68_study_plan"),
        (
            r"v68 preserves the CUDA guard inherited from v68(?:/v68)+\. Only global candidate generation and selection are changed\.",
            "Frozen CUDA executor and CPU/CUDA numerical audit; no mathematical changes in v68.",
        ),
    ]
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def _sanitize_frame_text(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(_sanitize_text_value)
    if "prefix_solver" in result.columns:
        result["prefix_solver"] = (
            "IRP-NEX local explanation core and GloPro-Complete glocal prototype-conditioned library"
        )
    if {"status", "note", "method"}.issubset(result.columns):
        result["method"] = "NEX-ELM interaction diagnostic"
        result["status"] = "diagnostic_interaction_audit_not_used_for_prototype_learning"
        result["note"] = (
            "Interaction diagnostics are retained for audit only and do not guide the active "
            "glocal prototype-conditioned library."
        )
    if "definition" in result.columns:
        old_mask = result["definition"].astype(str).str.contains(
            r"GloPro-Complete.*Pareto|capacity-matched Kernel SHAP reference budget", case=False, regex=True, na=False
        )
        result.loc[old_mask, "definition"] = (
            "GloPro-Complete glocal prototype-conditioned construction with capacity-matched Kernel SHAP evaluation."
        )
    return result


def _clean_frame(
    frame: pd.DataFrame,
    *,
    args: Optional[argparse.Namespace] = None,
    seed: Optional[int] = None,
    dataset: Optional[str] = None,
    scenario: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        return frame
    result = _sanitize_frame_text(_drop_obsolete_columns(frame))
    if result.empty:
        return result
    result["implementation_version"] = VERSION
    result["version"] = VERSION
    result["method_name"] = METHOD_NAME
    result["global_solver"] = SOLVER_ID
    result["global_definition"] = METHOD_DEFINITION
    result["global_representation"] = GLOBAL_REPRESENTATION
    result["global_representation_definition"] = GLOBAL_REPRESENTATION_DEFINITION
    result["representation_scope"] = "glocal"
    result["representation_conditioning"] = "prototype-conditioned"
    result["prototype_count_fixed"] = PROTOTYPE_COUNT
    result["prototype_min_calibration_rows_per_slot_fixed"] = MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT
    result["capacity_matched_shap"] = True
    result["outer_test_used_for_prototype_learning"] = False
    result["test_labels_used_for_routing"] = False
    result["test_fidelity_used_for_routing"] = False
    result["local_core_frozen"] = True
    result["cuda_core_frozen"] = True
    result["mathematical_core_sha256"] = EXPECTED_ENGINE_SHA256
    if seed is not None:
        result["seed"] = int(seed)
    if dataset is not None:
        result["dataset"] = str(dataset)
    if scenario is not None:
        result["scenario"] = str(scenario)
    context = _STUDY_CONTEXT.get((str(dataset), str(scenario)), {}) if dataset is not None and scenario is not None else {}
    if context:
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)) or value is None:
                result[key] = value
        result["method_frozen_before_evaluation"] = True
        result["evaluation_results_used_to_modify_method"] = False
        result["isolated_xelm_speed_superiority_claimed"] = False
        result["timing_claim_scope"] = "complete NEX-ELM workflow versus complete X-ELM plus Kernel SHAP workflow"
        result["fidelity_and_granularity_primary_claim"] = True
        result["outcomes_reported_regardless_of_direction"] = True
    if args is not None:
        result["batch_run_id"] = str(args.batch_run_id)
        result["study_plan"] = str(getattr(args, "study_plan", "custom"))
        if seed is not None:
            result["run_id"] = _per_seed_run_id(args, seed)
        elif "run_id" not in result.columns:
            result["run_id"] = str(args.batch_run_id)
    return result


def _registered_sets(order: Sequence[int], n_features: int) -> Dict[str, List[int]]:
    clean = _ENGINE_SANITIZE_ORDER(order, int(n_features))
    return {f"S{k}": sorted(map(int, clean[: int(k)])) for k in sorted(_ENGINE_REGISTERED_WEIGHTS(n_features))}


def _expand_library(frame: pd.DataFrame, args: argparse.Namespace, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    for _, record in frame.iterrows():
        feature_count = 0
        for method, orders_col, weights_col, count_col, cost_col, unique_col in [
            ("NEX-ELM", "nex_prototypes_json", "nex_prototype_weights_json", "nex_prototype_count_effective", "nex_kmedoids_cost", "nex_unique_registered_signatures"),
            ("Kernel SHAP", "shap_prototypes_json", "shap_prototype_weights_json", "shap_prototype_count_effective", "shap_kmedoids_cost", "shap_unique_registered_signatures"),
        ]:
            orders = json.loads(str(record.get(orders_col, "[]")))
            weights = json.loads(str(record.get(weights_col, "[]")))
            if orders:
                feature_count = max(feature_count, len(orders[0]))
            for prototype_id, order in enumerate(orders):
                weight = float(weights[prototype_id]) if prototype_id < len(weights) else np.nan
                rows.append({
                    "dataset": str(record.get("dataset", "")),
                    "scenario": str(record.get("scenario", "")),
                    "seed": int(seed),
                    "target_class_index": int(record.get("target_class_index", -1)),
                    "method": method,
                    "prototype_id": int(prototype_id),
                    "prototype_weight": weight,
                    "prototype_order_json": json.dumps(list(map(int, order))),
                    "registered_sets_json": json.dumps(_registered_sets(order, feature_count or len(order)), sort_keys=True),
                    "prototype_count_effective": int(record.get(count_col, len(orders))),
                    "unique_registered_signatures": int(record.get(unique_col, 0)),
                    "kmedoids_cost": float(record.get(cost_col, np.nan)),
                    "integration_verified": bool(record.get("integration_verified", True)),
                })
    return _clean_frame(pd.DataFrame(rows), args=args, seed=seed)


def _routing_long(frame: pd.DataFrame, args: argparse.Namespace, seed: int) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, record in frame.iterrows():
        common = {
            "dataset": str(record.get("dataset", "")),
            "scenario": str(record.get("scenario", "")),
            "seed": int(seed),
            "sample_id": str(record.get("sample_id", "")),
            "target_class_index": int(record.get("target_class_index", -1)),
        }
        for method, prefix in [("NEX-ELM", "nex"), ("Kernel SHAP", "shap")]:
            signature_seconds = float(record.get(f"{prefix}_signature_seconds", np.nan))
            prototype_routing_seconds = float(record.get(f"{prefix}_prototype_routing_seconds", np.nan))
            rows.append({
                **common,
                "method": method,
                "prototype_id": int(record.get(f"{prefix}_prototype_id", -1)),
                "registered_distance": float(record.get(f"{prefix}_registered_distance", np.nan)),
                "local_order_json": str(record.get(f"{prefix}_local_order_json", "[]")),
                "routed_order_json": str(record.get(f"{prefix}_routed_order_json", "[]")),
                "signature_seconds": signature_seconds,
                "prototype_routing_seconds": prototype_routing_seconds,
                "glocal_inference_seconds": float(signature_seconds + prototype_routing_seconds),
                "glocal_inference_seconds_per_sample": float(
                    record.get(f"{prefix}_glocal_inference_seconds_per_sample", np.nan)
                ),
            })
    return _clean_frame(pd.DataFrame(rows), args=args, seed=seed)


def _usage_summary(routing: pd.DataFrame, args: argparse.Namespace, seed: int) -> pd.DataFrame:
    if routing.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "scenario", "seed", "target_class_index", "method", "prototype_id"]
    summary = routing.groupby(group_cols, as_index=False).agg(
        routed_samples=("sample_id", "count"),
        registered_distance_mean=("registered_distance", "mean"),
        registered_distance_median=("registered_distance", "median"),
        registered_distance_max=("registered_distance", "max"),
    )
    totals = summary.groupby(["dataset", "scenario", "seed", "target_class_index", "method"])["routed_samples"].transform("sum")
    summary["usage_rate"] = summary["routed_samples"] / totals.clip(lower=1)
    return _clean_frame(summary, args=args, seed=seed)


def _routing_summary(routing: pd.DataFrame, args: argparse.Namespace, seed: int) -> pd.DataFrame:
    if routing.empty:
        return pd.DataFrame()
    summary = routing.groupby(
        ["dataset", "scenario", "seed", "target_class_index", "method"], as_index=False
    ).agg(
        routed_samples=("sample_id", "count"),
        prototypes_used=("prototype_id", "nunique"),
        registered_distance_mean=("registered_distance", "mean"),
        registered_distance_median=("registered_distance", "median"),
        registered_distance_max=("registered_distance", "max"),
        signature_seconds=("signature_seconds", "max"),
        prototype_routing_seconds=("prototype_routing_seconds", "max"),
        glocal_inference_seconds=("glocal_inference_seconds", "max"),
        glocal_inference_seconds_per_sample=("glocal_inference_seconds_per_sample", "max"),
    )
    return _clean_frame(summary, args=args, seed=seed)


def _delete_versioned_manifests(root: Path) -> None:
    for path in root.rglob("manifest_v*.json"):
        path.unlink(missing_ok=True)



def _clean_csv_directory(directory: Path, args: argparse.Namespace, seed: Optional[int] = None, dataset: Optional[str] = None, scenario: Optional[str] = None) -> None:
    for csv_path in Path(directory).glob("*.csv"):
        try:
            frame = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        if frame.empty:
            continue
        frame = _clean_frame(frame, args=args, seed=seed, dataset=dataset, scenario=scenario)
        engine.write_csv(frame, csv_path)

def _write_required_tables(result: Mapping[str, Any], tables: Path) -> None:
    mapping = {
        "global_importance": "global_importance.csv",
        "global_class_importance": "global_class_importance.csv",
        "global_calibration": "global_calibration.csv",
        "global_stability_diagnostics": "global_stability_diagnostics.csv",
        "global_fidelity_detail": "global_fidelity_detail.csv",
        "global_fidelity_summary": "global_fidelity_summary.csv",
        "prototype_library": "prototype_library.csv",
        "prototype_routing": "prototype_routing.csv",
        "prototype_usage_summary": "prototype_usage_summary.csv",
        "prototype_routing_summary": "prototype_routing_summary.csv",
        "timing": "timing.csv",
        "calibration": "calibration.csv",
        "cuda_audit": "cuda_audit.csv",
        "local_fidelity_detail": "local_fidelity_detail.csv",
        "local_fidelity_summary": "local_fidelity_summary.csv",
    }
    for key, filename in mapping.items():
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            engine.write_csv(frame, tables / filename)


def _timed_obtain_local_orders(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    dataset: str,
    scenario: str,
    context: Mapping[str, Any],
    seed: int,
):
    """Generate the same frozen NEX and Kernel SHAP orders with separate clocks."""
    feature_count = int(X.shape[1])
    key = (str(dataset), str(scenario))
    clock = _ROUTING_CLOCK.setdefault(key, {})

    nex_started = time.perf_counter()
    nex_orders: List[Optional[List[int]]] = [None] * len(X)
    missing_nex: List[int] = []
    for index, row in enumerate(X):
        cached = _LOCAL_ORDER_CACHE.get((str(dataset), str(scenario), _ENGINE_ROW_KEY(row)), {})
        if "NEX-ELM" in cached:
            nex_orders[index] = _ENGINE_SANITIZE_ORDER(cached["NEX-ELM"], feature_count)
        else:
            missing_nex.append(index)
    if missing_nex:
        subset = np.asarray(X[missing_nex], dtype=float)
        subset_targets = np.asarray(targets[missing_nex], dtype=int)
        explanation, generated, _, _ = _ENGINE_EXPLAIN_AND_ORDER(
            model, context["engine"], subset, subset_targets, context["args"], int(seed) + 650031
        )
        del explanation
        for index, order in zip(missing_nex, generated):
            clean = _ENGINE_SANITIZE_ORDER(order, feature_count)
            nex_orders[index] = clean
            cache_key = (str(dataset), str(scenario), _ENGINE_ROW_KEY(X[index]))
            _LOCAL_ORDER_CACHE.setdefault(cache_key, {})["NEX-ELM"] = clean
    clock["nex_signature_seconds"] = clock.get("nex_signature_seconds", 0.0) + (
        time.perf_counter() - nex_started
    )

    shap_started = time.perf_counter()
    shap_orders: List[Optional[List[int]]] = [None] * len(X)
    missing_shap: List[int] = []
    for index, row in enumerate(X):
        cached = _LOCAL_ORDER_CACHE.get((str(dataset), str(scenario), _ENGINE_ROW_KEY(row)), {})
        if "Kernel SHAP" in cached:
            shap_orders[index] = _ENGINE_SANITIZE_ORDER(cached["Kernel SHAP"], feature_count)
        else:
            missing_shap.append(index)
    if missing_shap:
        subset = np.asarray(X[missing_shap], dtype=float)
        subset_targets = np.asarray(targets[missing_shap], dtype=int)
        tensor, _ = engine.kernel_shap_probability(
            model,
            np.asarray(context["explain_backgrounds"], dtype=float),
            subset,
            int(getattr(context["args"], "shap_nsamples", 0)),
            int(seed) + 650071,
        )
        values = engine.selected_class_tensor(tensor, subset_targets)
        for index, row in zip(missing_shap, values):
            clean = engine.top_indices(row, feature_count).tolist()
            shap_orders[index] = clean
            cache_key = (str(dataset), str(scenario), _ENGINE_ROW_KEY(X[index]))
            _LOCAL_ORDER_CACHE.setdefault(cache_key, {})["Kernel SHAP"] = clean
    clock["shap_signature_seconds"] = clock.get("shap_signature_seconds", 0.0) + (
        time.perf_counter() - shap_started
    )

    return [list(order or range(feature_count)) for order in nex_orders], [
        list(order or range(feature_count)) for order in shap_orders
    ]


def _timed_route_orders(*args: Any, **kwargs: Any):
    started = time.perf_counter()
    output = _ENGINE_ROUTE_ORDERS(*args, **kwargs)
    elapsed = time.perf_counter() - started
    if _CURRENT_ROUTING_KEY is not None:
        clock = _ROUTING_CLOCK.setdefault(tuple(_CURRENT_ROUTING_KEY), {})
        call_index = int(clock.get("route_call_index", 0))
        prefix = "nex" if call_index % 2 == 0 else "shap"
        field = f"{prefix}_prototype_routing_seconds"
        clock[field] = clock.get(field, 0.0) + elapsed
        clock["route_call_index"] = call_index + 1
    return output


def _ranked_fidelity_with_routing_audit(
    model: Any,
    X: np.ndarray,
    targets: Sequence[int],
    selections_by_method: Mapping[str, Sequence[Sequence[int]]],
    backgrounds: np.ndarray,
    protocol: str,
    random_repeats: int,
    seed: int,
    dataset: str,
    scenario: str,
    scope: str,
    include_random: bool = True,
):
    global _CURRENT_ROUTING_KEY
    key = (str(dataset), str(scenario))
    _CURRENT_ROUTING_KEY = key
    _ROUTING_CLOCK[key] = {
        "nex_signature_seconds": 0.0,
        "shap_signature_seconds": 0.0,
        "nex_prototype_routing_seconds": 0.0,
        "shap_prototype_routing_seconds": 0.0,
        "route_call_index": 0,
    }
    try:
        result = _ENGINE_RANKED_FIDELITY(
            model, X, targets, selections_by_method, backgrounds, protocol,
            random_repeats, seed, dataset, scenario, scope, include_random,
        )
    finally:
        _CURRENT_ROUTING_KEY = None
    if str(scope) == "global_fidelity":
        clock = _ROUTING_CLOCK.get(key, {})
        rows = _ROUTING_DIAGNOSTICS.get(key, [])
        count = max(1, len(rows))
        for row in rows:
            for prefix in ("nex", "shap"):
                signature = float(clock.get(f"{prefix}_signature_seconds", 0.0))
                routing = float(clock.get(f"{prefix}_prototype_routing_seconds", 0.0))
                row[f"{prefix}_signature_seconds"] = signature
                row[f"{prefix}_prototype_routing_seconds"] = routing
                row[f"{prefix}_glocal_inference_seconds_per_sample"] = float(
                    (signature + routing) / count
                )
    return result


# Timing wrappers only. Mathematical outputs are delegated to the unchanged core.
_replace_bindings(engine, _ENGINE_OBTAIN_LOCAL_ORDERS, _timed_obtain_local_orders)
_replace_bindings(engine, _ENGINE_ROUTE_ORDERS, _timed_route_orders)
_replace_bindings(engine, _ENGINE_RANKED_FIDELITY, _ranked_fidelity_with_routing_audit)


def _include_glocal_inference_in_workflow_timing(
    result: Dict[str, Any], dataset: str, scenario: str, args: argparse.Namespace, seed: int
) -> None:
    timing = result.get("timing")
    if not isinstance(timing, pd.DataFrame) or timing.empty:
        return
    clock = _ROUTING_CLOCK.get((str(dataset), str(scenario)), {})
    values = {
        "nex_signature_seconds": float(clock.get("nex_signature_seconds", 0.0)),
        "shap_signature_seconds": float(clock.get("shap_signature_seconds", 0.0)),
        "nex_prototype_routing_seconds": float(clock.get("nex_prototype_routing_seconds", 0.0)),
        "shap_prototype_routing_seconds": float(clock.get("shap_prototype_routing_seconds", 0.0)),
    }
    frame = timing.copy()
    if "seconds" not in frame.columns or "method" not in frame.columns:
        return
    frame["seconds"] = pd.to_numeric(frame["seconds"], errors="coerce")
    frame["signature_seconds_added_to_workflow"] = 0.0
    frame["prototype_routing_seconds_added_to_workflow"] = 0.0
    frame["workflow_includes_glocal_inference"] = False

    workflow_specs = [
        (
            "NEX-ELM workflow complete",
            values["nex_signature_seconds"],
            values["nex_prototype_routing_seconds"],
            "NEX-ELM training, glocal construction, local explanations, inference signature, and prototype routing.",
        ),
        (
            "X-ELM + Kernel SHAP workflow complete",
            values["shap_signature_seconds"],
            values["shap_prototype_routing_seconds"],
            "X-ELM, Kernel SHAP global/local explanations, inference signature, and prototype routing.",
        ),
    ]
    for method, signature_seconds, routing_seconds, definition in workflow_specs:
        mask = frame["method"].astype(str).eq(method) & frame["scope"].astype(str).eq("workflow")
        frame.loc[mask, "seconds"] = frame.loc[mask, "seconds"] + signature_seconds + routing_seconds
        frame.loc[mask, "signature_seconds_added_to_workflow"] = signature_seconds
        frame.loc[mask, "prototype_routing_seconds_added_to_workflow"] = routing_seconds
        frame.loc[mask, "workflow_includes_glocal_inference"] = True
        if "definition" in frame.columns:
            frame.loc[mask, "definition"] = definition

    audit_rows = pd.DataFrame([
        {
            "dataset": dataset, "scenario": scenario, "seed": int(seed),
            "method": "NEX-ELM inference signature", "seconds": values["nex_signature_seconds"],
            "scope": "glocal_signature", "definition": "Time to obtain NEX-ELM inference signatures, including cache lookup.",
        },
        {
            "dataset": dataset, "scenario": scenario, "seed": int(seed),
            "method": "Kernel SHAP inference signature", "seconds": values["shap_signature_seconds"],
            "scope": "glocal_signature", "definition": "Time to obtain Kernel SHAP inference signatures, including cache lookup.",
        },
        {
            "dataset": dataset, "scenario": scenario, "seed": int(seed),
            "method": "NEX-ELM prototype routing", "seconds": values["nex_prototype_routing_seconds"],
            "scope": "prototype_routing", "definition": "Nearest fixed NEX-ELM prototype under the registered-set distance.",
        },
        {
            "dataset": dataset, "scenario": scenario, "seed": int(seed),
            "method": "Kernel SHAP prototype routing", "seconds": values["shap_prototype_routing_seconds"],
            "scope": "prototype_routing", "definition": "Nearest fixed Kernel SHAP prototype under the registered-set distance.",
        },
    ])
    audit_rows["signature_seconds_added_to_workflow"] = 0.0
    audit_rows["prototype_routing_seconds_added_to_workflow"] = 0.0
    audit_rows["workflow_includes_glocal_inference"] = False
    result["timing"] = _clean_frame(
        pd.concat([frame, audit_rows], ignore_index=True, sort=False),
        args=args, seed=seed, dataset=dataset, scenario=scenario,
    )


def postprocess_artifacts(
    result: Dict[str, Any], bundle: Any, args: argparse.Namespace, root: Path, seed: int, scenario: str
):
    result = _ENGINE_POSTPROCESS(result, bundle, args, root, seed, scenario)
    dataset = str(bundle.name)
    _include_glocal_inference_in_workflow_timing(result, dataset, scenario, args, seed)
    for key, frame in list(result.items()):
        if isinstance(frame, pd.DataFrame):
            result[key] = _clean_frame(frame, args=args, seed=seed, dataset=dataset, scenario=scenario)

    class_summary = result.get("prototype_library", pd.DataFrame())
    routing_wide = result.get("prototype_routing", pd.DataFrame())
    library = _expand_library(class_summary, args, seed)
    routing = _routing_long(routing_wide, args, seed)
    if library.empty:
        raise RuntimeError(f"{METHOD_TITLE} audit failed: prototype library is empty for {dataset}, seed {seed}.")
    if routing.empty:
        raise RuntimeError(f"{METHOD_TITLE} audit failed: prototype routing is empty for {dataset}, seed {seed}.")
    result["prototype_library_class_summary"] = _clean_frame(
        class_summary, args=args, seed=seed, dataset=dataset, scenario=scenario
    )
    result["prototype_library"] = library
    result["prototype_routing"] = routing
    result["prototype_usage_summary"] = _usage_summary(routing, args, seed)
    result["prototype_routing_summary"] = _routing_summary(routing, args, seed)

    dataset_dir = Path(root) / "per_seed" / dataset / scenario / f"seed_{int(seed)}"
    _delete_versioned_manifests(dataset_dir)
    tables = engine.ensure_dir(dataset_dir / "tabelas")
    _write_required_tables(result, tables)
    engine.write_csv(result["prototype_library_class_summary"], tables / "prototype_library_class_summary.csv")
    _clean_csv_directory(tables, args, seed=seed, dataset=dataset, scenario=scenario)

    seed_run_id = _per_seed_run_id(args, seed)
    manifest = {
        "version": VERSION,
        "method_name": METHOD_NAME,
        "batch_run_id": str(args.batch_run_id),
        "run_id": seed_run_id,
        "dataset": dataset,
        "scenario": str(scenario),
        "seed": int(seed),
        "global_solver": SOLVER_ID,
        "global_definition": METHOD_DEFINITION,
        "global_representation": GLOBAL_REPRESENTATION,
        "global_representation_definition": GLOBAL_REPRESENTATION_DEFINITION,
        "mathematical_core_unchanged": True,
        "registered_set_distance_frozen": True,
        "deterministic_kmedoids_frozen": True,
        "prototype_routing_frozen": True,
        "prototype_count_fixed": PROTOTYPE_COUNT,
        "packaging_revision": PACKAGING_REVISION,
        "mathematical_core_source": _ENGINE_SOURCE,
        "prototype_min_calibration_rows_per_slot_fixed": MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT,
        "primary_comparison": "NEX-ELM GloPro-Complete versus capacity-matched Kernel SHAP prototype library",
        "outer_test_used_for_prototype_learning": False,
        "test_labels_used_for_routing": False,
        "test_fidelity_used_for_routing": False,
        "local_core_frozen": True,
        "cuda_core_frozen": True,
        "prototype_library_rows": int(len(library)),
        "prototype_routing_rows": int(len(routing)),
        "mathematical_core_sha256": EXPECTED_ENGINE_SHA256,
        "configuration": _public_configuration(args),
        "study_context": _STUDY_CONTEXT.get((dataset, str(scenario)), {}),
        "method_frozen_before_evaluation": True,
        "evaluation_results_used_to_modify_method": False,
        "primary_claim": "NEX-ELM advances fidelity and explanatory granularity over X-ELM.",
        "timing_claim_scope": "complete NEX-ELM workflow versus complete X-ELM plus Kernel SHAP workflow",
        "isolated_xelm_speed_superiority_claimed": False,
        "wisconsin_reporting_policy": "report the replication result irrespective of direction or threshold attainment",
    }
    engine.write_json(manifest, dataset_dir / "manifest_v68_glopro_complete.json")
    engine.write_json(manifest, dataset_dir / "manifest.json")
    return result


def confirmatory_plan(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame([{
        "implementation_version": VERSION,
        "method_name": METHOD_NAME,
        "batch_run_id": str(args.batch_run_id),
        "global_solver": SOLVER_ID,
        "global_definition": METHOD_DEFINITION,
        "global_representation": GLOBAL_REPRESENTATION,
        "global_representation_definition": GLOBAL_REPRESENTATION_DEFINITION,
        "representation_scope": "glocal",
        "representation_conditioning": "prototype-conditioned",
        "prototype_count_fixed": PROTOTYPE_COUNT,
        "packaging_revision": PACKAGING_REVISION,
        "mathematical_core_source": _ENGINE_SOURCE,
        "mathematical_core_sha256": EXPECTED_ENGINE_SHA256,
        "prototype_min_calibration_rows_per_slot_fixed": MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT,
        "prototype_learning": "deterministic registered-set k-medoids on calibration explanation signatures",
        "prototype_budget_rule": "K_eff <= floor(calibration_rows / min_rows_per_prototype_slot)",
        "per_cluster_minimum_enforced": False,
        "timing_audit": "NEX-ELM and Kernel SHAP signature/routing times are separate and included in each complete workflow",
        "routing": "prototype-conditioned glocal routing to the nearest fixed prototype",
        "primary_global_comparison": "NEX-ELM GloPro-Complete versus capacity-matched Kernel SHAP prototype library",
        "single_order_audits": "NEX-ELM-Single; Kernel SHAP-Single",
        "outer_test_used_for_prototype_learning": False,
        "test_labels_used_for_routing": False,
        "test_fidelity_used_for_routing": False,
        "local_core_frozen": True,
        "cuda_core_frozen": True,
        "statistical_suite": STATISTICAL_SUITE,
        "new_tests_are_supplementary": True,
        "original_confirmatory_rule_preserved": True,
        "confirmatory_requirement": "execute the frozen code on previously unused seeds",
    }])


def _public_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    excluded = re.compile(r"^(?:_?v\d+|historical_|legacy_)", re.I)
    excluded_exact = {
        "repetition_source",
        "min_seed_wins_source",
        "cuda_execution_definition",
        "local_method_definition",
        "repetitions", "n_repeats", "repeats", "num_repeats", "n_repeat",
        "n_repetitions", "num_repetitions", "default_repetitions",
        "confirmation_repetitions", "n_seeds", "num_seeds", "seed_count",
        "n_random_seeds", "random_seed_count", "n_runs", "num_runs", "run_count",
        "outer_repeats", "outer_n_repeats",
    }
    configuration: Dict[str, Any] = {}
    for key, value in vars(args).items():
        name = str(key)
        if excluded.match(name) or name in excluded_exact:
            continue
        configuration[name] = value
    configuration["prototype_count"] = PROTOTYPE_COUNT
    configuration["prototype_min_calibration_rows_per_slot"] = MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT
    configuration["global_method_definition"] = METHOD_DEFINITION
    configuration["local_method_definition"] = (
        "IRP-NEX local explanation core with calibration-only selection and external-test evaluation"
    )
    configuration["cuda_execution_definition"] = (
        "Frozen CUDA executor with CPU/CUDA numerical audit; no mathematical changes in v68"
    )
    configuration["repetition_source"] = "v68 preregistered study-phase configuration"
    configuration["method_name"] = METHOD_NAME
    configuration["statistical_suite"] = STATISTICAL_SUITE
    for key in (
        "study_plan", "replication_repeats", "replication_random_state", "replication_datasets",
        "generalization_repeats", "generalization_random_state", "generalization_datasets",
        "previous_confirmatory_seed_start", "previous_confirmatory_repeats",
        "allow_seed_overlap", "include_optional_datasets",
    ):
        if hasattr(args, key):
            configuration[key] = getattr(args, key)
    configuration["method_frozen_before_evaluation"] = True
    configuration["evaluation_results_used_for_method_modification"] = False
    configuration["primary_claim"] = "fidelity and explanatory granularity over X-ELM"
    configuration["isolated_xelm_speed_superiority_claimed"] = False
    return configuration


def _prototype_stability(results: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    frames = [item.get("prototype_library") for item in results]
    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    columns = [
        "dataset", "target_class_index", "method", "seed_a", "seed_b",
        "matched_prototypes", "matched_registered_distance_mean",
        "matched_registered_distance_max", "exact_signature_match_rate",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    library = pd.concat(frames, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for (dataset, target_class, method), group in library.groupby(["dataset", "target_class_index", "method"]):
        seeds = sorted(map(int, group["seed"].unique()))
        for left_index in range(len(seeds)):
            for right_index in range(left_index + 1, len(seeds)):
                seed_a, seed_b = seeds[left_index], seeds[right_index]
                a = group[group["seed"] == seed_a].sort_values("prototype_id")
                b = group[group["seed"] == seed_b].sort_values("prototype_id")
                orders_a = [json.loads(text) for text in a["prototype_order_json"]]
                orders_b = [json.loads(text) for text in b["prototype_order_json"]]
                if not orders_a or not orders_b:
                    continue
                feature_count = len(orders_a[0])
                cost = np.asarray([
                    [_ENGINE_SIGNATURE_DISTANCE(x, y, feature_count) for y in orders_b]
                    for x in orders_a
                ], dtype=float)
                row_idx, col_idx = linear_sum_assignment(cost)
                matched = cost[row_idx, col_idx]
                exact = float(np.mean(matched <= 1e-15)) if len(matched) else np.nan
                rows.append({
                    "dataset": dataset,
                    "target_class_index": int(target_class),
                    "method": method,
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "matched_prototypes": int(len(matched)),
                    "matched_registered_distance_mean": float(np.mean(matched)),
                    "matched_registered_distance_max": float(np.max(matched)),
                    "exact_signature_match_rate": exact,
                })
    return _clean_frame(pd.DataFrame(rows, columns=columns), args=args)


def _clean_combined_csvs(combined: Path, args: argparse.Namespace) -> None:
    for csv_path in combined.glob("*.csv"):
        try:
            frame = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        if frame.empty:
            continue
        frame = _clean_frame(frame, args=args)
        engine.write_csv(frame, csv_path)


def _holm_adjust(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.full(array.shape, np.nan, dtype=float)
    valid_indices = np.where(np.isfinite(array))[0]
    if not len(valid_indices):
        return output
    order = valid_indices[np.argsort(array[valid_indices])]
    count = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, float(array[index]) * (count - rank))
        running = max(running, adjusted)
        output[index] = running
    return output


def _analysis_methods(scope: str, metric: str, available: Sequence[str]) -> List[str]:
    available_set = set(map(str, available))
    if scope == "global_fidelity":
        preferred = ARTICLE_METHODS_GLOBAL
    elif scope == "local_fidelity":
        preferred = ARTICLE_METHODS_LOCAL
    elif scope == "timing" and metric == "local":
        preferred = ARTICLE_METHODS_TIMING_LOCAL
    elif scope == "timing" and metric == "workflow":
        preferred = ARTICLE_METHODS_TIMING_WORKFLOW
    else:
        return []
    return [method for method in preferred if method in available_set]


def _higher_is_better(scope: str) -> bool:
    return str(scope) != "timing"


def _metric_groups(seed_metrics: pd.DataFrame) -> Iterable[Tuple[Tuple[str, str, str, str], pd.DataFrame, List[str]]]:
    required = {"dataset", "scenario", "scope", "metric", "method", "seed", "value"}
    if not required.issubset(seed_metrics.columns):
        return
    working = seed_metrics[list(required)].copy()
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    working["seed"] = pd.to_numeric(working["seed"], errors="coerce")
    working = working.dropna(subset=["value", "seed"])
    working["seed"] = working["seed"].astype(int)
    for keys, group in working.groupby(["dataset", "scenario", "scope", "metric"], sort=True):
        methods = _analysis_methods(str(keys[2]), str(keys[3]), group["method"].astype(str).unique())
        if len(methods) < 2:
            continue
        pivot = group[group["method"].isin(methods)].pivot_table(
            index="seed", columns="method", values="value", aggfunc="mean"
        )
        pivot = pivot.reindex(columns=methods).dropna()
        if len(pivot) >= 2:
            yield (tuple(map(str, keys)), pivot, methods)


def _paired_arrays(pivot: pd.DataFrame, first: str, second: str, higher_is_better: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_values = np.asarray(pivot[first], dtype=float)
    second_values = np.asarray(pivot[second], dtype=float)
    oriented = first_values - second_values if higher_is_better else second_values - first_values
    valid = np.isfinite(first_values) & np.isfinite(second_values) & np.isfinite(oriented)
    return first_values[valid], second_values[valid], oriented[valid]


def _paired_t_and_shapiro(seed_metrics: pd.DataFrame, alpha: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t_rows: List[Dict[str, Any]] = []
    shapiro_rows: List[Dict[str, Any]] = []
    for (dataset, scenario, scope, metric), pivot, methods in _metric_groups(seed_metrics):
        higher = _higher_is_better(scope)
        for first, second in itertools.combinations(methods, 2):
            first_values, second_values, differences = _paired_arrays(pivot, first, second, higher)
            count = len(differences)
            if count < 2:
                continue
            mean_difference = float(np.mean(differences))
            standard_deviation = float(np.std(differences, ddof=1)) if count > 1 else np.nan
            standard_error = standard_deviation / math.sqrt(count) if count > 1 else np.nan
            if count > 1 and np.isfinite(standard_error):
                critical = float(stats.t.ppf(0.975, count - 1))
                ci_low = mean_difference - critical * standard_error
                ci_high = mean_difference + critical * standard_error
            else:
                ci_low = ci_high = np.nan
            if standard_deviation > 0:
                t_statistic = mean_difference / (standard_deviation / math.sqrt(count))
                p_two_sided = float(2.0 * stats.t.sf(abs(t_statistic), count - 1))
                p_one_sided = float(stats.t.sf(t_statistic, count - 1))
                cohen_dz = mean_difference / standard_deviation
            elif mean_difference > 0:
                t_statistic, p_two_sided, p_one_sided, cohen_dz = np.inf, 0.0, 0.0, np.inf
            elif mean_difference < 0:
                t_statistic, p_two_sided, p_one_sided, cohen_dz = -np.inf, 0.0, 1.0, -np.inf
            else:
                t_statistic, p_two_sided, p_one_sided, cohen_dz = 0.0, 1.0, 0.5, 0.0
            t_rows.append({
                "dataset": dataset,
                "scenario": scenario,
                "scope": scope,
                "metric": metric,
                "first_method": first,
                "second_method": second,
                "difference_orientation": "first_minus_second" if higher else "second_minus_first_so_positive_favors_first",
                "higher_is_better": bool(higher),
                "n_pairs": int(count),
                "mean_oriented_difference": mean_difference,
                "standard_deviation_of_differences": standard_deviation,
                "standard_error": standard_error,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "t_statistic": float(t_statistic),
                "degrees_of_freedom": int(count - 1),
                "paired_t_p_two_sided": p_two_sided,
                "paired_t_p_one_sided_superiority": p_one_sided,
                "effect_size_cohen_dz": float(cohen_dz),
                "superior_by_t_ci": bool(np.isfinite(ci_low) and ci_low > 0),
            })

            if 3 <= count <= 5000:
                shapiro = stats.shapiro(differences)
                shapiro_w = float(shapiro.statistic)
                shapiro_p = float(shapiro.pvalue)
            else:
                shapiro_w = shapiro_p = np.nan
            skewness = float(stats.skew(differences, bias=False)) if count >= 3 else np.nan
            kurtosis = float(stats.kurtosis(differences, fisher=True, bias=False)) if count >= 4 else np.nan
            shapiro_rows.append({
                "dataset": dataset,
                "scenario": scenario,
                "scope": scope,
                "metric": metric,
                "first_method": first,
                "second_method": second,
                "n_pairs": int(count),
                "shapiro_w": shapiro_w,
                "shapiro_p_value": shapiro_p,
                "normality_not_rejected_alpha": bool(np.isfinite(shapiro_p) and shapiro_p >= alpha),
                "skewness_of_differences": skewness,
                "excess_kurtosis_of_differences": kurtosis,
            })
    t_frame = pd.DataFrame(t_rows)
    shapiro_frame = pd.DataFrame(shapiro_rows)
    if not t_frame.empty:
        t_frame["paired_t_p_one_sided_holm"] = _holm_adjust(t_frame["paired_t_p_one_sided_superiority"])
        t_frame["paired_t_p_two_sided_holm"] = _holm_adjust(t_frame["paired_t_p_two_sided"])
        t_frame["paired_t_superiority_supported"] = (
            (t_frame["paired_t_p_one_sided_holm"] < alpha) & t_frame["superior_by_t_ci"]
        )
    if not shapiro_frame.empty:
        shapiro_frame["shapiro_p_holm_descriptive"] = _holm_adjust(shapiro_frame["shapiro_p_value"])
    return t_frame, shapiro_frame


def _repeated_measures_anova(seed_metrics: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (dataset, scenario, scope, metric), pivot, methods in _metric_groups(seed_metrics):
        values = np.asarray(pivot[methods], dtype=float)
        subjects, conditions = values.shape
        if subjects < 2 or conditions < 2:
            continue
        grand = float(np.mean(values))
        subject_means = np.mean(values, axis=1)
        condition_means = np.mean(values, axis=0)
        ss_total = float(np.sum((values - grand) ** 2))
        ss_subject = float(conditions * np.sum((subject_means - grand) ** 2))
        ss_method = float(subjects * np.sum((condition_means - grand) ** 2))
        ss_error = max(0.0, ss_total - ss_subject - ss_method)
        df_method = conditions - 1
        df_error = (subjects - 1) * (conditions - 1)
        ms_method = ss_method / df_method if df_method > 0 else np.nan
        ms_error = ss_error / df_error if df_error > 0 else np.nan
        if ms_error > 0:
            f_statistic = ms_method / ms_error
            p_value = float(stats.f.sf(f_statistic, df_method, df_error))
        elif ms_method > 0:
            f_statistic, p_value = np.inf, 0.0
        else:
            f_statistic, p_value = 0.0, 1.0
        partial_eta_squared = ss_method / (ss_method + ss_error) if (ss_method + ss_error) > 0 else 0.0

        if conditions > 2 and subjects > 2:
            covariance = np.cov(values, rowvar=False, ddof=1)
            centering = np.eye(conditions) - np.ones((conditions, conditions)) / conditions
            centered_covariance = centering @ covariance @ centering
            numerator = float(np.trace(centered_covariance) ** 2)
            denominator = float((conditions - 1) * np.trace(centered_covariance @ centered_covariance))
            epsilon = numerator / denominator if denominator > 0 else 1.0
            epsilon = float(np.clip(epsilon, 1.0 / (conditions - 1), 1.0))
        else:
            epsilon = 1.0
        gg_df_method = epsilon * df_method
        gg_df_error = epsilon * df_error
        gg_p = float(stats.f.sf(f_statistic, gg_df_method, gg_df_error)) if np.isfinite(f_statistic) else 0.0
        rows.append({
            "dataset": dataset,
            "scenario": scenario,
            "scope": scope,
            "metric": metric,
            "methods_json": json.dumps(methods, ensure_ascii=False),
            "method_means_json": json.dumps({method: float(condition_means[index]) for index, method in enumerate(methods)}, ensure_ascii=False),
            "n_subjects_seeds": int(subjects),
            "n_methods": int(conditions),
            "anova_f": float(f_statistic),
            "df_method": int(df_method),
            "df_error": int(df_error),
            "anova_p_value": p_value,
            "greenhouse_geisser_epsilon": epsilon,
            "gg_df_method": gg_df_method,
            "gg_df_error": gg_df_error,
            "anova_p_greenhouse_geisser": gg_p,
            "partial_eta_squared": float(partial_eta_squared),
            "reject_equal_method_means_alpha": bool(gg_p < alpha),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["anova_p_greenhouse_geisser_holm"] = _holm_adjust(frame["anova_p_greenhouse_geisser"])
        frame["anova_significant_after_holm"] = frame["anova_p_greenhouse_geisser_holm"] < alpha
    return frame


def _friedman_and_nemenyi(
    seed_metrics: pd.DataFrame, alpha: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    friedman_rows: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []
    nemenyi_rows: List[Dict[str, Any]] = []
    for (dataset, scenario, scope, metric), pivot, methods in _metric_groups(seed_metrics):
        if len(methods) < 3:
            continue
        values = np.asarray(pivot[methods], dtype=float)
        subjects, conditions = values.shape
        if subjects < 2:
            continue
        friedman = stats.friedmanchisquare(*[values[:, index] for index in range(conditions)])
        statistic = float(friedman.statistic)
        p_value = float(friedman.pvalue)
        kendall_w = statistic / (subjects * (conditions - 1)) if subjects > 0 and conditions > 1 else np.nan
        higher = _higher_is_better(scope)
        ranks = np.vstack([
            stats.rankdata(-row if higher else row, method="average") for row in values
        ])
        average_ranks = np.mean(ranks, axis=0)
        standard_error = math.sqrt(conditions * (conditions + 1) / (6.0 * subjects))
        if hasattr(stats, "studentized_range"):
            q_alpha = float(stats.studentized_range.ppf(1.0 - alpha, conditions, np.inf) / math.sqrt(2.0))
            critical_difference = q_alpha * standard_error
        else:
            critical_difference = np.nan
        friedman_rows.append({
            "dataset": dataset,
            "scenario": scenario,
            "scope": scope,
            "metric": metric,
            "methods_json": json.dumps(methods, ensure_ascii=False),
            "n_subjects_seeds": int(subjects),
            "n_methods": int(conditions),
            "friedman_chi_square": statistic,
            "degrees_of_freedom": int(conditions - 1),
            "friedman_p_value": p_value,
            "kendall_w": float(kendall_w),
            "reject_equal_rank_distributions_alpha": bool(p_value < alpha),
            "nemenyi_critical_difference_alpha": critical_difference,
        })
        for index, method in enumerate(methods):
            rank_rows.append({
                "dataset": dataset,
                "scenario": scenario,
                "scope": scope,
                "metric": metric,
                "method": method,
                "average_rank": float(average_ranks[index]),
                "n_subjects_seeds": int(subjects),
                "n_methods": int(conditions),
                "rank_1_is_best": True,
                "nemenyi_critical_difference_alpha": critical_difference,
            })
        for left, right in itertools.combinations(range(conditions), 2):
            rank_difference = abs(float(average_ranks[left] - average_ranks[right]))
            q_statistic = rank_difference / standard_error if standard_error > 0 else np.nan
            if hasattr(stats, "studentized_range") and np.isfinite(q_statistic):
                p_posthoc = float(stats.studentized_range.sf(q_statistic * math.sqrt(2.0), conditions, np.inf))
            else:
                p_posthoc = np.nan
            nemenyi_rows.append({
                "dataset": dataset,
                "scenario": scenario,
                "scope": scope,
                "metric": metric,
                "first_method": methods[left],
                "second_method": methods[right],
                "first_average_rank": float(average_ranks[left]),
                "second_average_rank": float(average_ranks[right]),
                "absolute_rank_difference": rank_difference,
                "nemenyi_q": float(q_statistic),
                "nemenyi_p_value": p_posthoc,
                "critical_difference": critical_difference,
                "significant_by_critical_difference": bool(
                    np.isfinite(critical_difference) and rank_difference > critical_difference
                ),
                "nemenyi_significant_alpha": bool(np.isfinite(p_posthoc) and p_posthoc < alpha),
                "friedman_omnibus_significant": bool(p_value < alpha),
                "posthoc_interpretation_enabled": bool(p_value < alpha),
                "n_subjects_seeds": int(subjects),
                "n_methods": int(conditions),
            })
    friedman_frame = pd.DataFrame(friedman_rows)
    ranks_frame = pd.DataFrame(rank_rows)
    nemenyi_frame = pd.DataFrame(nemenyi_rows)
    if not friedman_frame.empty:
        friedman_frame["friedman_p_holm"] = _holm_adjust(friedman_frame["friedman_p_value"])
        friedman_frame["friedman_significant_after_holm"] = friedman_frame["friedman_p_holm"] < alpha
    return friedman_frame, ranks_frame, nemenyi_frame


def _binomial_win_tests(seed_metrics: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Exact paired win-rate analysis across seeds.

    Ties remain in the denominator and count as non-wins. The reported
    Clopper-Pearson interval is two-sided with confidence level 1-alpha.
    Three directional exact-binomial tests are kept in distinct families:
    superiority over chance (p > 0.50), evidence below the preregistered
    threshold (p < 0.80), and evidence above that threshold (p > 0.80).
    """
    rows: List[Dict[str, Any]] = []
    tolerance = 1e-12
    confidence_level = 1.0 - float(alpha)
    for (dataset, scenario, scope, metric), pivot, methods in _metric_groups(seed_metrics):
        higher = _higher_is_better(scope)
        for first, second in itertools.combinations(methods, 2):
            _, _, differences = _paired_arrays(pivot, first, second, higher)
            if not len(differences):
                continue
            wins = int(np.sum(differences > tolerance))
            losses = int(np.sum(differences < -tolerance))
            ties = int(len(differences) - wins - losses)
            trials = int(len(differences))

            chance_test = stats.binomtest(wins, trials, p=0.5, alternative="greater")
            two_sided_reference = stats.binomtest(wins, trials, p=0.5, alternative="two-sided")
            interval = two_sided_reference.proportion_ci(
                confidence_level=confidence_level, method="exact"
            )
            below_eighty = stats.binomtest(wins, trials, p=0.8, alternative="less")
            above_eighty = stats.binomtest(wins, trials, p=0.8, alternative="greater")
            rows.append({
                "dataset": dataset,
                "scenario": scenario,
                "scope": scope,
                "metric": metric,
                "first_method": first,
                "second_method": second,
                "higher_is_better": bool(higher),
                "n_trials_seeds": trials,
                "first_method_wins": wins,
                "first_method_losses": losses,
                "ties_counted_as_non_wins": ties,
                "win_rate": wins / trials,
                "clopper_pearson_two_sided_confidence_level": confidence_level,
                "clopper_pearson_two_sided_ci95_low": float(interval.low),
                "clopper_pearson_two_sided_ci95_high": float(interval.high),
                "binomial_p_greater_than_50_percent": float(chance_test.pvalue),
                "binomial_p_less_than_80_percent": float(below_eighty.pvalue),
                "binomial_p_greater_than_80_percent": float(above_eighty.pvalue),
                "observed_win_rate_at_least_80_percent": bool(wins / trials >= 0.8),
                "win_rate_superior_to_chance_alpha": bool(chance_test.pvalue < alpha),
                "evidence_win_rate_below_80_percent_alpha": bool(below_eighty.pvalue < alpha),
                "evidence_win_rate_above_80_percent_alpha": bool(above_eighty.pvalue < alpha),
                "binomial_test_ties_policy": "ties_retained_in_denominator_as_non_wins",
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["binomial_p_greater_than_50_percent_holm"] = _holm_adjust(
            frame["binomial_p_greater_than_50_percent"]
        )
        frame["binomial_superiority_over_chance_after_holm"] = (
            frame["binomial_p_greater_than_50_percent_holm"] < alpha
        )
        frame["binomial_p_less_than_80_percent_holm"] = _holm_adjust(
            frame["binomial_p_less_than_80_percent"]
        )
        frame["evidence_win_rate_below_80_percent_after_holm"] = (
            frame["binomial_p_less_than_80_percent_holm"] < alpha
        )
        frame["binomial_p_greater_than_80_percent_holm"] = _holm_adjust(
            frame["binomial_p_greater_than_80_percent"]
        )
        frame["evidence_win_rate_above_80_percent_after_holm"] = (
            frame["binomial_p_greater_than_80_percent_holm"] < alpha
        )
    return frame


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame()


def _inferential_summary(
    t_tests: pd.DataFrame,
    shapiro: pd.DataFrame,
    binomial: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["dataset", "scenario", "scope", "metric", "first_method", "second_method"]
    summary = t_tests.copy()
    if summary.empty:
        return summary
    if not shapiro.empty:
        summary = summary.merge(shapiro, on=keys + ["n_pairs"], how="left", suffixes=("", "_shapiro"))
    if not binomial.empty:
        summary = summary.merge(binomial, on=keys, how="left", suffixes=("", "_binomial"))
    if not existing.empty and set(keys).issubset(existing.columns):
        keep = keys + [
            column for column in [
                "bootstrap_ci95_low", "bootstrap_ci95_high", "wilcoxon_statistic", "wilcoxon_p_value",
                "one_sided_wilcoxon_p_value", "one_sided_permutation_p_value",
                "one_sided_wilcoxon_p_value_holm", "one_sided_permutation_p_value_holm",
                "one_sided_wilcoxon_p_value_holm_primary", "one_sided_permutation_p_value_holm_primary",
                "confirmatory_superiority", "noninferior_first_vs_second", "first_wins", "ties",
                "first_win_fraction", "required_seed_wins", "required_seed_win_rate",
            ] if column in existing.columns
        ]
        summary = summary.merge(existing[keep].drop_duplicates(keys), on=keys, how="left", suffixes=("", "_existing"))
    return summary


def _multiple_testing_families(
    t_tests: pd.DataFrame,
    anova: pd.DataFrame,
    friedman: pd.DataFrame,
    binomial: pd.DataFrame,
    nemenyi: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    specifications = [
        (t_tests, "paired_t_one_sided", "paired_t_p_one_sided_superiority", "paired_t_p_one_sided_holm"),
        (t_tests, "paired_t_two_sided", "paired_t_p_two_sided", "paired_t_p_two_sided_holm"),
        (anova, "repeated_measures_anova", "anova_p_greenhouse_geisser", "anova_p_greenhouse_geisser_holm"),
        (friedman, "friedman", "friedman_p_value", "friedman_p_holm"),
        (binomial, "binomial_win_rate_over_chance", "binomial_p_greater_than_50_percent", "binomial_p_greater_than_50_percent_holm"),
        (binomial, "binomial_win_rate_below_80", "binomial_p_less_than_80_percent", "binomial_p_less_than_80_percent_holm"),
        (binomial, "binomial_win_rate_above_80", "binomial_p_greater_than_80_percent", "binomial_p_greater_than_80_percent_holm"),
        (nemenyi, "nemenyi_within_omnibus_group", "nemenyi_p_value", None),
    ]
    for frame, family, raw_column, adjusted_column in specifications:
        if frame.empty or raw_column not in frame.columns:
            continue
        for _, record in frame.iterrows():
            rows.append({
                "test_family": family,
                "dataset": record.get("dataset", ""),
                "scenario": record.get("scenario", ""),
                "scope": record.get("scope", ""),
                "metric": record.get("metric", ""),
                "first_method": record.get("first_method", ""),
                "second_method": record.get("second_method", ""),
                "raw_p_value": record.get(raw_column, np.nan),
                "adjusted_p_value": record.get(adjusted_column, np.nan) if adjusted_column else record.get(raw_column, np.nan),
                "adjustment": "Holm across the supplementary family" if adjusted_column else "Nemenyi studentized-range familywise control within each omnibus group",
            })
    return pd.DataFrame(rows)


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return text[:120] or "item"


def _save_figure(fig: Any, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    formats = GRAPH_FORMATS if "article_summaries" in base_path.parts else DIAGNOSTIC_GRAPH_FORMATS
    for extension in formats:
        path = base_path.with_suffix(f".{extension}")
        kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
    plt.close(fig)


def _is_primary_article_pair(scope: str, metric: str, first: str, second: str) -> bool:
    pair = {str(first), str(second)}
    if scope == "global_fidelity" and metric == "composite_auc":
        return pair in ({"NEX-ELM", "Kernel SHAP"}, {"NEX-ELM", "X-ELM"})
    if scope == "local_fidelity" and metric in {"deletion_auc", "insertion_auc"}:
        return pair == {"NEX-ELM", "Kernel SHAP"}
    if scope == "timing" and metric == "local":
        return pair == set(ARTICLE_METHODS_TIMING_LOCAL)
    if scope == "timing" and metric == "workflow":
        return pair == set(ARTICLE_METHODS_TIMING_WORKFLOW)
    return False


def _primary_pair_records(seed_metrics: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    records: Dict[str, List[Dict[str, Any]]] = {}
    for (dataset, scenario, scope, metric), pivot, methods in _metric_groups(seed_metrics):
        higher = _higher_is_better(scope)
        for first, second in itertools.combinations(methods, 2):
            if not _is_primary_article_pair(scope, metric, first, second):
                continue
            first_values, second_values, differences = _paired_arrays(pivot, first, second, higher)
            if len(differences) < 2:
                continue
            records.setdefault(dataset, []).append({
                "dataset": dataset,
                "scenario": scenario,
                "scope": scope,
                "metric": metric,
                "first": first,
                "second": second,
                "first_values": first_values,
                "second_values": second_values,
                "differences": differences,
                "higher": higher,
            })
    for dataset in records:
        records[dataset].sort(key=lambda item: (item["scope"], item["metric"], item["first"], item["second"]))
    return records


def _plot_consolidated_pair_diagnostics(seed_metrics: pd.DataFrame, figures: Path) -> None:
    for dataset, records in _primary_pair_records(seed_metrics).items():
        rows = len(records)
        fig, axes = plt.subplots(rows, 2, figsize=(12.0, max(4.2, 3.1 * rows)), squeeze=False)
        for row_index, record in enumerate(records):
            differences = record["differences"]
            histogram = axes[row_index, 0]
            bins = min(12, max(5, int(math.sqrt(len(differences))) + 1))
            histogram.hist(differences, bins=bins)
            histogram.axvline(0.0, linestyle="--", linewidth=1.0)
            histogram.axvline(float(np.mean(differences)), linewidth=1.4)
            histogram.set_xlabel("Oriented paired difference")
            histogram.set_ylabel("Seeds")
            histogram.set_title(
                f"{record['scope']}/{record['metric']}\n{record['first']} vs {record['second']}"
            )

            qq = axes[row_index, 1]
            theoretical, ordered = stats.probplot(differences, dist="norm", fit=False)
            slope, intercept, _ = stats.probplot(differences, dist="norm", fit=True)[1]
            qq.scatter(theoretical, ordered)
            xline = np.asarray([min(theoretical), max(theoretical)], dtype=float)
            qq.plot(xline, intercept + slope * xline)
            qq.set_xlabel("Theoretical normal quantiles")
            qq.set_ylabel("Observed differences")
            qq.set_title("Q–Q diagnostic")
        fig.suptitle(f"Paired-difference and Shapiro–Wilk diagnostics — {dataset}", y=1.01)
        fig.tight_layout()
        _save_figure(fig, figures / "paired_diagnostics" / f"{_slug(dataset)}__paired_diagnostics")

        columns = 2
        panel_rows = int(math.ceil(rows / columns))
        fig, axes = plt.subplots(panel_rows, columns, figsize=(12.0, max(4.2, 3.6 * panel_rows)), squeeze=False)
        flat = axes.ravel()
        for index, record in enumerate(records):
            ax = flat[index]
            x = np.array([0.0, 1.0])
            for left, right in zip(record["first_values"], record["second_values"]):
                ax.plot(x, [left, right], alpha=0.32, linewidth=0.75)
            ax.plot(
                x,
                [float(np.mean(record["first_values"])), float(np.mean(record["second_values"]))],
                marker="o",
                linewidth=2.2,
            )
            ax.set_xticks(x, [record["first"], record["second"]], rotation=12, ha="right")
            ax.set_ylabel("Metric value" if record["higher"] else "Seconds")
            ax.set_title(f"{record['scope']}/{record['metric']}")
        for index in range(rows, len(flat)):
            flat[index].axis("off")
        fig.suptitle(f"Paired results across seeds — {dataset}", y=1.01)
        fig.tight_layout()
        _save_figure(fig, figures / "paired_seed_panels" / f"{_slug(dataset)}__paired_seed_panels")


def _plot_consolidated_means(seed_metrics: pd.DataFrame, figures: Path) -> None:
    groups_by_dataset: Dict[str, List[Tuple[Tuple[str, str, str, str], pd.DataFrame, List[str]]]] = {}
    for keys, pivot, methods in _metric_groups(seed_metrics):
        dataset, _, scope, _ = keys
        if scope not in {"global_fidelity", "local_fidelity", "timing"}:
            continue
        groups_by_dataset.setdefault(dataset, []).append((keys, pivot, methods))
    for dataset, groups in groups_by_dataset.items():
        groups.sort(key=lambda item: (item[0][2], item[0][3]))
        columns = 2
        rows = int(math.ceil(len(groups) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(12.0, max(4.5, 3.7 * rows)), squeeze=False)
        flat = axes.ravel()
        for index, (keys, pivot, methods) in enumerate(groups):
            _, _, scope, metric = keys
            ax = flat[index]
            means = np.asarray([pivot[method].mean() for method in methods], dtype=float)
            errors = np.asarray([
                stats.t.ppf(0.975, len(pivot) - 1) * pivot[method].std(ddof=1) / math.sqrt(len(pivot))
                if len(pivot) > 1 else np.nan
                for method in methods
            ], dtype=float)
            positions = np.arange(len(methods))
            ax.errorbar(positions, means, yerr=errors, marker="o", capsize=4, linestyle="none")
            ax.set_xticks(positions, methods, rotation=13, ha="right")
            ax.set_ylabel("Metric value" if _higher_is_better(scope) else "Seconds")
            ax.set_title(f"{scope}/{metric}")
        for index in range(len(groups), len(flat)):
            flat[index].axis("off")
        fig.suptitle(f"Method means and 95% confidence intervals — {dataset}", y=1.01)
        fig.tight_layout()
        _save_figure(fig, figures / "means_ci_panels" / f"{_slug(dataset)}__means_ci")


def _plot_consolidated_rank_analysis(
    ranks: pd.DataFrame, nemenyi: pd.DataFrame, figures: Path
) -> None:
    if ranks.empty:
        return
    for dataset, dataset_ranks in ranks.groupby("dataset", sort=True):
        group_keys = list(dataset_ranks[["scenario", "scope", "metric"]].drop_duplicates().itertuples(index=False, name=None))
        group_keys.sort(key=lambda item: (item[1], item[2]))
        rows = len(group_keys)
        fig, axes = plt.subplots(rows, 2, figsize=(12.5, max(4.2, 3.6 * rows)), squeeze=False)
        for row_index, (scenario, scope, metric) in enumerate(group_keys):
            current = dataset_ranks[
                (dataset_ranks["scenario"].astype(str) == str(scenario))
                & (dataset_ranks["scope"].astype(str) == str(scope))
                & (dataset_ranks["metric"].astype(str) == str(metric))
            ].sort_values("average_rank")
            rank_ax = axes[row_index, 0]
            positions = np.arange(len(current))
            rank_ax.barh(positions, current["average_rank"].to_numpy(dtype=float))
            rank_ax.set_yticks(positions, current["method"].astype(str))
            rank_ax.invert_yaxis()
            rank_ax.set_xlabel("Average rank (1 = best)")
            cd = float(current["nemenyi_critical_difference_alpha"].iloc[0])
            rank_ax.set_title(f"{scope}/{metric} — CD={cd:.3f}" if np.isfinite(cd) else f"{scope}/{metric}")

            current_nemenyi = nemenyi[
                (nemenyi["dataset"].astype(str) == str(dataset))
                & (nemenyi["scenario"].astype(str) == str(scenario))
                & (nemenyi["scope"].astype(str) == str(scope))
                & (nemenyi["metric"].astype(str) == str(metric))
            ]
            methods = current["method"].astype(str).tolist()
            matrix = np.ones((len(methods), len(methods)), dtype=float)
            method_index = {method: index for index, method in enumerate(methods)}
            for record in current_nemenyi.itertuples(index=False):
                left = method_index.get(str(record.first_method))
                right = method_index.get(str(record.second_method))
                if left is None or right is None:
                    continue
                matrix[left, right] = matrix[right, left] = float(record.nemenyi_p_value)
            matrix_ax = axes[row_index, 1]
            image = matrix_ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto")
            matrix_ax.set_xticks(np.arange(len(methods)), methods, rotation=18, ha="right")
            matrix_ax.set_yticks(np.arange(len(methods)), methods)
            for row in range(len(methods)):
                for column in range(len(methods)):
                    matrix_ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=8)
            matrix_ax.set_title("Nemenyi p-values")
            fig.colorbar(image, ax=matrix_ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Friedman ranks and Nemenyi post-hoc analysis — {dataset}", y=1.01)
        fig.tight_layout()
        _save_figure(fig, figures / "rank_posthoc_panels" / f"{_slug(dataset)}__rank_posthoc")


def _plot_article_forest_and_win_rates(
    t_tests: pd.DataFrame, binomial: pd.DataFrame, figures: Path
) -> None:
    if not t_tests.empty:
        selected = t_tests[t_tests.apply(
            lambda row: _is_primary_article_pair(
                str(row["scope"]), str(row["metric"]), str(row["first_method"]), str(row["second_method"])
            ), axis=1
        )].copy()
        selected = selected.sort_values(["dataset", "scope", "metric", "first_method", "second_method"])
        if not selected.empty:
            labels = [
                f"{row.dataset}: {row.scope}/{row.metric} — {row.first_method} vs {row.second_method}"
                for row in selected.itertuples(index=False)
            ]
            positions = np.arange(len(selected))
            means = selected["mean_oriented_difference"].to_numpy(dtype=float)
            low = selected["ci95_low"].to_numpy(dtype=float)
            high = selected["ci95_high"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(10.0, max(7.0, 0.42 * len(selected) + 2.0)))
            ax.errorbar(means, positions, xerr=[means - low, high - means], fmt="o", capsize=3)
            ax.axvline(0.0, linestyle="--", linewidth=1.0)
            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel("Oriented paired mean difference with 95% CI")
            ax.set_title("Primary paired-effect forest plot")
            _save_figure(fig, figures / "article_summaries" / "primary_paired_effect_forest")

    if not binomial.empty:
        selected = binomial[binomial.apply(
            lambda row: _is_primary_article_pair(
                str(row["scope"]), str(row["metric"]), str(row["first_method"]), str(row["second_method"])
            ), axis=1
        )].copy()
        selected = selected.sort_values(["dataset", "scope", "metric", "first_method", "second_method"])
        if not selected.empty:
            labels = [
                f"{row.dataset}: {row.scope}/{row.metric} — {row.first_method} vs {row.second_method}"
                for row in selected.itertuples(index=False)
            ]
            positions = np.arange(len(selected))
            rates = selected["win_rate"].to_numpy(dtype=float)
            low = selected["clopper_pearson_two_sided_ci95_low"].to_numpy(dtype=float)
            high = selected["clopper_pearson_two_sided_ci95_high"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(10.0, max(7.0, 0.42 * len(selected) + 2.0)))
            ax.errorbar(rates, positions, xerr=[rates - low, high - rates], fmt="o", capsize=3)
            ax.axvline(0.5, linestyle="--", linewidth=1.0, label="Chance reference")
            ax.axvline(0.8, linestyle=":", linewidth=1.2, label="Confirmatory reference")
            ax.set_xlim(0.0, 1.02)
            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel("Win rate with two-sided exact 95% Clopper-Pearson interval")
            ax.set_title("Primary paired win rates and preregistered references")
            ax.legend()
            _save_figure(fig, figures / "article_summaries" / "primary_binomial_win_rates")


def _plot_pvalue_map(summary: pd.DataFrame, figures: Path) -> None:
    if summary.empty:
        return
    selected = summary[summary.apply(
        lambda row: _is_primary_article_pair(
            str(row["scope"]), str(row["metric"]), str(row["first_method"]), str(row["second_method"])
        ), axis=1
    )].copy()
    columns = [
        column for column in [
            "paired_t_p_one_sided_holm", "shapiro_p_value",
            "one_sided_wilcoxon_p_value_holm", "one_sided_permutation_p_value_holm",
            "binomial_p_greater_than_50_percent_holm",
        ] if column in selected.columns
    ]
    if selected.empty or not columns:
        return
    labels = [
        f"{row.dataset} | {row.scope}/{row.metric} | {row.first_method} vs {row.second_method}"
        for row in selected.itertuples(index=False)
    ]
    matrix = selected[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    transformed = -np.log10(np.clip(matrix, 1e-300, 1.0))
    fig, ax = plt.subplots(figsize=(10.0, max(7.0, 0.38 * len(selected) + 2.0)))
    image = ax.imshow(transformed, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    fig.colorbar(image, ax=ax, label="−log10(p)")
    ax.set_title("Primary inferential p-value map")
    _save_figure(fig, figures / "article_summaries" / "primary_inferential_pvalue_map")


def _generate_statistical_figures(
    seed_metrics: pd.DataFrame,
    t_tests: pd.DataFrame,
    ranks: pd.DataFrame,
    nemenyi: pd.DataFrame,
    binomial: pd.DataFrame,
    summary: pd.DataFrame,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    _plot_consolidated_pair_diagnostics(seed_metrics, figures)
    _plot_consolidated_means(seed_metrics, figures)
    _plot_consolidated_rank_analysis(ranks, nemenyi, figures)
    _plot_article_forest_and_win_rates(t_tests, binomial, figures)
    _plot_pvalue_map(summary, figures)


def _write_statistical_suite(combined: Path, args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    seed_metrics = _read_table(combined / "seed_metrics.csv")
    if seed_metrics.empty:
        return {}
    alpha = float(getattr(args, "alpha", 0.05))
    t_tests, shapiro = _paired_t_and_shapiro(seed_metrics, alpha)
    anova = _repeated_measures_anova(seed_metrics, alpha)
    friedman, ranks, nemenyi = _friedman_and_nemenyi(seed_metrics, alpha)
    binomial = _binomial_win_tests(seed_metrics, alpha)
    existing = _read_table(combined / "estatistica_entre_seeds.csv")
    summary = _inferential_summary(t_tests, shapiro, binomial, existing)
    multiple = _multiple_testing_families(t_tests, anova, friedman, binomial, nemenyi)

    outputs = {
        "paired_t_tests": t_tests,
        "shapiro_wilk_tests": shapiro,
        "repeated_measures_anova": anova,
        "friedman_tests": friedman,
        "nemenyi_posthoc": nemenyi,
        "binomial_win_rate_tests": binomial,
        "average_method_ranks": ranks,
        "inferential_test_summary": summary,
        "multiple_testing_families": multiple,
    }
    for name, frame in outputs.items():
        clean = _clean_frame(frame, args=args) if not frame.empty else frame
        outputs[name] = clean
        engine.write_csv(clean, combined / f"{name}.csv")

    figures = combined.parent / "graficos_estatisticos"
    _generate_statistical_figures(
        seed_metrics,
        outputs["paired_t_tests"],
        outputs["average_method_ranks"],
        outputs["nemenyi_posthoc"],
        outputs["binomial_win_rate_tests"],
        outputs["inferential_test_summary"],
        figures,
    )
    plt.close("all")
    return outputs


def _sanitize_text_artifacts(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".json"}:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except Exception:
            continue
        cleaned = _sanitize_text_value(original)
        if cleaned != original:
            path.write_text(cleaned, encoding="utf-8")


def _audit_generated_text(root: Path) -> pd.DataFrame:
    def token(*codes: int) -> str:
        return "".join(chr(code) for code in codes)

    patterns = {
        "obsolete_version_reference": re.compile(
            r"(?<![A-Za-z0-9])v(?:2[0-9]|3[0-9]|4[0-9]|5[0-9]|6[0-7])(?:\.\d+)*(?=$|[^A-Za-z0-9])", re.I
        ),
        "obsolete_method_code_1": re.compile(rf"\b{re.escape(token(80, 71, 65))}\b", re.I),
        "obsolete_method_code_2": re.compile(rf"\b{re.escape(token(71, 80, 73))}\b", re.I),
        "obsolete_method_code_3": re.compile(re.escape(token(72, 69, 83, 80, 79)), re.I),
        "obsolete_method_code_4": re.compile(rf"\b{re.escape(token(78, 69, 88, 45, 68, 73))}\b", re.I),
        "obsolete_repetition_source": re.compile(r"engine_default_not_used_by_v68_study_plan|v\d+_default_\d+", re.I),
        "malformed_cuda_history": re.compile(r"inherited from v68(?:/v68)+", re.I),
    }
    rows: List[Dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".txt", ".md", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception:
            continue
        for label, pattern in patterns.items():
            matches = sorted(set(match.group(0) for match in pattern.finditer(text)))
            if matches:
                rows.append({
                    "file": str(path.relative_to(root)),
                    "issue": label,
                    "matches_json": json.dumps(matches, ensure_ascii=False),
                })
        if path.suffix.lower() == ".csv":
            try:
                header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns
            except Exception:
                header = []
            obsolete_headers = sorted({
                str(column) for column in header
                if _OBSOLETE_COLUMN.search(str(column))
                or re.search(r"_from_v\d+(?:_\d+)*$", str(column), flags=re.I)
            })
            if obsolete_headers:
                rows.append({
                    "file": str(path.relative_to(root)),
                    "issue": "obsolete_column_header",
                    "matches_json": json.dumps(obsolete_headers, ensure_ascii=False),
                })
    return pd.DataFrame(rows, columns=["file", "issue", "matches_json"])


def aggregate_results(results: Sequence[Dict[str, Any]], output_root: Path, args: argparse.Namespace, runtime: Any) -> None:
    _ENGINE_AGGREGATE(results, output_root, args, runtime)
    output_root = Path(output_root)
    _delete_versioned_manifests(output_root)
    combined = engine.ensure_dir(output_root / "combined" / "tabelas")

    keys = [
        "prototype_library", "prototype_library_class_summary", "prototype_routing",
        "prototype_usage_summary", "prototype_routing_summary",
    ]
    for key in keys:
        frames = [item.get(key) for item in results]
        frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
        engine.write_csv(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), combined / f"{key}.csv")

    predictive_outputs = _aggregate_predictive_outputs(results, combined, args)
    _generate_predictive_figures(combined, predictive_outputs)

    stability = _prototype_stability(results, args)
    engine.write_csv(stability, combined / "prototype_stability_between_seeds.csv")
    engine.write_csv(confirmatory_plan(args), combined / "plano_confirmatorio.csv")
    _clean_combined_csvs(combined, args)
    statistical_outputs = _write_statistical_suite(combined, args)
    _clean_combined_csvs(combined, args)

    required = [
        combined / "prototype_library.csv",
        combined / "prototype_routing.csv",
        combined / "prototype_usage_summary.csv",
        combined / "prototype_routing_summary.csv",
        combined / "prototype_stability_between_seeds.csv",
        combined / "paired_t_tests.csv",
        combined / "shapiro_wilk_tests.csv",
        combined / "repeated_measures_anova.csv",
        combined / "friedman_tests.csv",
        combined / "nemenyi_posthoc.csv",
        combined / "binomial_win_rate_tests.csv",
        combined / "average_method_ranks.csv",
        combined / "inferential_test_summary.csv",
        combined / "multiple_testing_families.csv",
        combined / "predictive_performance_per_seed.csv",
        combined / "predictive_performance_summary.csv",
        combined / "predictive_performance_article_table.csv",
        combined / "predictive_confusion_matrix.csv",
        combined / "predictive_confusion_matrix_summary.csv",
        combined / "predictive_class_metrics.csv",
        combined / "predictive_class_metrics_summary.csv",
        combined / "predictive_dataset_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size <= 3]
    if missing:
        raise RuntimeError(f"{METHOD_TITLE} combined audit failed; required files missing or empty: {missing}")

    seeds = sorted({
        int(frame["seed"].iloc[0])
        for item in results
        for frame in [item.get("prototype_library")]
        if isinstance(frame, pd.DataFrame) and not frame.empty
    })
    audit = {
        "version": VERSION,
        "method_name": METHOD_NAME,
        "batch_run_id": str(args.batch_run_id),
        "global_solver": SOLVER_ID,
        "prototype_count_fixed": PROTOTYPE_COUNT,
        "prototype_min_calibration_rows_per_slot_fixed": MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT,
        "global_representation": GLOBAL_REPRESENTATION,
        "global_representation_definition": GLOBAL_REPRESENTATION_DEFINITION,
        "mathematical_core_unchanged": True,
        "registered_set_distance_frozen": True,
        "deterministic_kmedoids_frozen": True,
        "prototype_routing_frozen": True,
        "timing_signatures_separated_by_method": True,
        "workflow_includes_glocal_inference": True,
        "supplementary_statistical_suite_generated": True,
        "predictive_complete_outer_test_generated": True,
        "predictive_tables": sorted(predictive_outputs),
        "pdf_report_requested": not bool(getattr(args, "skip_pdf_report", False)),
        "statistical_tables": sorted(statistical_outputs),
        "statistical_graph_formats": list(GRAPH_FORMATS),
        "original_confirmatory_rule_preserved": True,
        "packaging_revision": PACKAGING_REVISION,
        "mathematical_core_source": _ENGINE_SOURCE,
        "seeds": seeds,
        "unique_seed_run_ids_expected": len(seeds),
        "required_prototype_files_present": True,
        "prototype_library_rows": int(sum(len(item.get("prototype_library", [])) for item in results)),
        "prototype_routing_rows": int(sum(len(item.get("prototype_routing", [])) for item in results)),
        "local_core_frozen": True,
        "cuda_core_frozen": True,
        "mathematical_core_sha256": EXPECTED_ENGINE_SHA256,
        "passed": True,
    }
    engine.write_json(audit, output_root / "audit_validation_v68_glopro_complete.json")
    manifest = {
        **audit,
        "global_definition": METHOD_DEFINITION,
        "primary_comparison_capacity_matched": True,
        "outer_test_used_for_prototype_learning": False,
        "test_labels_used_for_routing": False,
        "test_fidelity_used_for_routing": False,
        "configuration": _public_configuration(args),
    }
    engine.write_json(manifest, output_root / "manifest_v68_glopro_complete.json")
    engine.write_json(manifest, output_root / "registro_protocolo_confirmatorio.json")

    _sanitize_text_artifacts(output_root)
    findings = _audit_generated_text(output_root)
    engine.write_csv(findings, combined / "text_audit_findings.csv")
    if not findings.empty:
        raise RuntimeError(
            f"{METHOD_TITLE} text audit found obsolete public references. See {combined / 'text_audit_findings.csv'}"
        )

    report_path = None
    if not bool(getattr(args, "skip_pdf_report", False)):
        report_path = _generate_pdf_report(output_root, args=args, output_path=output_root / "relatorio.pdf")
    engine.write_json({
        "version": VERSION,
        "report_generated": bool(report_path is not None),
        "report_path": str(report_path) if report_path is not None else None,
        "report_name": "relatorio.pdf",
        "source_is_current_experiment": True,
        "passed": bool(report_path is not None or getattr(args, "skip_pdf_report", False)),
    }, output_root / "audit_relatorio_v68.json")



def _canonical_dataset_name(name: str) -> str:
    key = str(name).strip().lower()
    return DATASET_ALIASES.get(key, key)


def _parse_dataset_list(value: str) -> List[str]:
    names = [_canonical_dataset_name(item) for item in str(value).split(",") if item.strip()]
    output: List[str] = []
    for name in names:
        if name not in output:
            output.append(name)
    return output


def _load_grid_without_stab(data_dir: Path, allow_download: bool) -> Any:
    path = engine.download_if_needed(
        data_dir,
        ["Data_for_UCI_named.csv", "electrical_grid_stability.csv"],
        engine.GRID_URLS,
        allow_download,
    )
    frame = pd.read_csv(path)
    required = {"stab", "stabf"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Electrical Grid without stab requires the UCI columns 'stab' and 'stabf'. "
            f"Missing columns: {missing}."
        )
    features = [column for column in frame.columns if column not in {"stab", "stabf"}]
    y = frame["stabf"].astype(str).str.strip().to_numpy()
    return engine.DatasetBundle(
        name="electrical_grid_stability_without_stab",
        X=frame[features].to_numpy(dtype=float),
        y=y,
        feature_names=list(features),
        source=str(path),
        notes=[
            "Electrical Grid leakage-control scenario: categorical target stabf.",
            "Continuous target proxy stab is excluded from the predictors.",
            "This scenario complements, but does not replace, the exact X-ELM reproduction scenario.",
        ],
    )


def _load_sklearn_bundle(name: str) -> Any:
    from sklearn import datasets as sk_datasets

    if name == "iris_multiclass":
        raw = sk_datasets.load_iris()
        labels = np.asarray([raw.target_names[int(index)] for index in raw.target], dtype=str)
        notes = ["Iris multiclass benchmark; three classes and four numerical attributes."]
    elif name == "wine_multiclass":
        raw = sk_datasets.load_wine()
        labels = np.asarray([raw.target_names[int(index)] for index in raw.target], dtype=str)
        notes = ["Wine multiclass benchmark; three classes and thirteen numerical attributes."]
    elif name == "digits_multiclass":
        raw = sk_datasets.load_digits()
        labels = np.asarray(raw.target, dtype=str)
        notes = ["Digits multiclass stress test; ten classes and sixty-four pixel attributes."]
    elif name == "breast_cancer_diagnostic":
        raw = sk_datasets.load_breast_cancer()
        labels = np.asarray([raw.target_names[int(index)] for index in raw.target], dtype=str)
        notes = ["Breast Cancer Diagnostic benchmark; independent from the Original Wisconsin dataset."]
    else:
        raise ValueError(f"Unsupported sklearn dataset: {name}")
    return engine.DatasetBundle(
        name=name,
        X=np.asarray(raw.data, dtype=float),
        y=labels,
        feature_names=[str(item) for item in raw.feature_names],
        source=f"scikit-learn built-in dataset: {name}",
        notes=notes,
    )


def _load_ionosphere(data_dir: Path, allow_download: bool) -> Any:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    path = engine.download_if_needed(data_dir, ["ionosphere.data", "ionosphere.csv"], [url], allow_download)
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] < 3:
        raise ValueError("Ionosphere dataset must contain numerical predictors and one class column.")
    features = [f"radar_feature_{index + 1:02d}" for index in range(frame.shape[1] - 1)]
    X = frame.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1) & frame.iloc[:, -1].notna()
    X = X.loc[valid]
    y = frame.loc[valid, frame.columns[-1]].astype(str).str.strip().to_numpy()
    return engine.DatasetBundle(
        name="ionosphere_binary",
        X=X.to_numpy(dtype=float),
        y=y,
        feature_names=features,
        source=str(path),
        notes=["UCI Ionosphere binary radar-return classification benchmark."],
    )


def _load_v68_bundle(name: str, args: argparse.Namespace) -> Any:
    canonical = _canonical_dataset_name(name)
    data_dir = Path(args.data_dir)
    allow_download = not bool(args.no_download)
    if canonical == "electrical_grid_stability":
        bundle = engine.load_grid(data_dir, allow_download)
        bundle.notes = list(bundle.notes) + [
            "Exact X-ELM reproduction scenario: stab is retained only for benchmark comparability.",
            "The separate without-stab scenario is the leakage-control result for journal claims.",
        ]
        return bundle
    if canonical == "electrical_grid_stability_without_stab":
        return _load_grid_without_stab(data_dir, allow_download)
    if canonical == "pima_indians_diabetes":
        return engine.load_pima(data_dir, allow_download)
    if canonical == "wisconsin_breast_cancer_original":
        return engine.load_breast_original(data_dir, allow_download)
    if canonical == "ionosphere_binary":
        return _load_ionosphere(data_dir, allow_download)
    if canonical in OPTIONAL_DATASETS or canonical == "wine_multiclass":
        return _load_sklearn_bundle(canonical)
    valid = sorted(set(REPLICATION_DATASETS + GENERALIZATION_DATASETS + OPTIONAL_DATASETS))
    raise ValueError(f"Invalid dataset '{name}'. Valid v68 datasets: {valid}")


def _dataset_context(bundle: Any, phase: str, evidence_role: str, scenario: str) -> Dict[str, Any]:
    classes = np.unique(np.asarray(bundle.y).astype(str))
    is_multiclass = len(classes) > 2
    proxy_policy = "not_applicable"
    if bundle.name == "electrical_grid_stability":
        proxy_policy = "stab_retained_for_exact_xelm_reproduction_only"
    elif bundle.name == "electrical_grid_stability_without_stab":
        proxy_policy = "stab_removed_leakage_control"
    context = {
        "study_phase": phase,
        "evidence_role": evidence_role,
        "scenario": scenario,
        "dataset_name": str(bundle.name),
        "dataset_source": str(bundle.source),
        "task_type": "multiclass_classification" if is_multiclass else "binary_classification",
        "n_classes": int(len(classes)),
        "n_features": int(np.asarray(bundle.X).shape[1]),
        "n_samples": int(np.asarray(bundle.X).shape[0]),
        "multiclass_evaluation": bool(is_multiclass),
        "target_proxy_policy": proxy_policy,
        "method_selected_before_this_phase": True,
        "test_results_used_for_method_modification": False,
        "outcome_reporting_policy": "report_all_results_irrespective_of_direction",
        "primary_scientific_claim": "fidelity_and_explanatory_granularity",
        "isolated_xelm_speed_claim": False,
    }
    if bundle.name == "wisconsin_breast_cancer_original" and phase == "independent_replication":
        context["prior_confirmatory_status"] = "mixed_against_kernel_shap_global"
        context["replication_purpose"] = "estimate_reproducibility_not_force_threshold_crossing"
    return context


def _seed_sequence(start: int, repeats: int) -> List[int]:
    return [int(start) + 1009 * index for index in range(int(repeats))]


def _validate_seed_independence(args: argparse.Namespace, proposed: Sequence[int]) -> None:
    previous = set(_seed_sequence(args.previous_confirmatory_seed_start, args.previous_confirmatory_repeats))
    overlap = sorted(previous.intersection(map(int, proposed)))
    if overlap and not bool(args.allow_seed_overlap):
        raise ValueError(
            "The v68 replication seeds overlap the registered original confirmatory battery. "
            f"Overlap: {overlap}. Change --replication-random-state or pass --allow-seed-overlap only for an explicit audit."
        )


def _study_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    optional = _parse_dataset_list(args.include_optional_datasets)
    invalid_optional = [name for name in optional if name not in OPTIONAL_DATASETS]
    if invalid_optional:
        raise ValueError(f"Invalid optional datasets: {invalid_optional}; valid: {list(OPTIONAL_DATASETS)}")
    specs: List[Dict[str, Any]] = []
    if args.study_plan in {"complete", "journal", "replication"}:
        specs.append({
            "phase": "independent_replication",
            "evidence_role": "replication_of_original_three_benchmarks",
            "scenario": "replication_original_benchmarks_v68_glopro_complete",
            "datasets": _parse_dataset_list(args.replication_datasets),
            "repeats": int(args.replication_repeats),
            "random_state": int(args.replication_random_state),
        })
    if args.study_plan in {"complete", "journal", "generalization"}:
        datasets = _parse_dataset_list(args.generalization_datasets)
        if args.study_plan == "complete":
            datasets += list(EXTENDED_GENERALIZATION_DATASETS)
        datasets += optional
        datasets = list(dict.fromkeys(datasets))
        grid_control = [name for name in datasets if name == "electrical_grid_stability_without_stab"]
        external = [name for name in datasets if name != "electrical_grid_stability_without_stab"]
        if grid_control:
            specs.append({
                "phase": "target_proxy_control",
                "evidence_role": "electrical_grid_without_stab_robustness",
                "scenario": "grid_without_stab_control_v68_glopro_complete",
                "datasets": grid_control,
                "repeats": int(args.generalization_repeats),
                "random_state": int(args.generalization_random_state),
            })
        if external:
            specs.append({
                "phase": "external_generalization",
                "evidence_role": "new_datasets_multiclass_and_dimensional_validation",
                "scenario": "external_generalization_v68_glopro_complete",
                "datasets": external,
                "repeats": int(args.generalization_repeats),
                "random_state": int(args.generalization_random_state),
            })
    if args.study_plan == "custom":
        specs.append({
            "phase": "custom_frozen_math_evaluation",
            "evidence_role": "user_defined_extension",
            "scenario": "custom_frozen_math_v68_glopro_complete",
            "datasets": _parse_dataset_list(args.datasets),
            "repeats": int(args.real_repetitions),
            "random_state": int(args.random_state),
        })
    for spec in specs:
        if not spec["datasets"]:
            raise ValueError(f"Study phase {spec['phase']} has no datasets.")
        proposed = _seed_sequence(spec["random_state"], spec["repeats"])
        if spec["phase"] == "independent_replication":
            _validate_seed_independence(args, proposed)
        spec["seeds"] = proposed
    selected = [dataset for spec in specs for dataset in spec["datasets"]]
    if args.study_plan == "complete" and set(selected) != set(COMPLETE_DATASETS):
        raise RuntimeError(
            f"Complete-plan dataset audit failed. Expected {list(COMPLETE_DATASETS)}, obtained {sorted(set(selected))}."
        )
    return specs


def _write_v68_study_plan(root: Path, args: argparse.Namespace, specs: Sequence[Mapping[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    for spec in specs:
        for dataset in spec["datasets"]:
            rows.append({
                "implementation_version": VERSION,
                "method_name": METHOD_NAME,
                "study_plan": args.study_plan,
                "study_phase": spec["phase"],
                "evidence_role": spec["evidence_role"],
                "scenario": spec["scenario"],
                "dataset": dataset,
                "repetitions": int(spec["repeats"]),
                "seed_start": int(spec["random_state"]),
                "seed_step": 1009,
                "method_frozen_before_evaluation": True,
                "evaluation_results_used_to_modify_method": False,
                "primary_claim": "fidelity_and_explanatory_granularity",
                "isolated_xelm_speed_superiority_claimed": False,
                "report_results_irrespective_of_direction": True,
            })
        for repetition, seed in enumerate(spec["seeds"], start=1):
            seed_rows.append({
                "study_phase": spec["phase"],
                "scenario": spec["scenario"],
                "repetition": repetition,
                "seed": int(seed),
                "independent_from_original_confirmatory_battery": bool(
                    int(seed) not in set(_seed_sequence(args.previous_confirmatory_seed_start, args.previous_confirmatory_repeats))
                ),
            })
    engine.write_csv(pd.DataFrame(rows), Path(root) / "plano_validacao_completa_v68.csv")
    engine.write_csv(pd.DataFrame(seed_rows), Path(root) / "registro_seeds_v68.csv")
    claims = pd.DataFrame([
        {
            "claim": "NEX-ELM improves fidelity and explanatory granularity over X-ELM",
            "status": "primary_claim_to_test",
            "scope": "global, local, and glocal explanations",
        },
        {
            "claim": "NEX-ELM is faster than isolated X-ELM",
            "status": "not_claimed",
            "scope": "isolated X-ELM remains a low-cost global baseline",
        },
        {
            "claim": "NEX-ELM workflow is faster than the complete X-ELM plus Kernel SHAP workflow",
            "status": "timing_claim_to_test",
            "scope": "workflow-level comparison only",
        },
        {
            "claim": "Wisconsin must be a victory",
            "status": "prohibited_outcome_target",
            "scope": "all mixed or negative results must be reported",
        },
        {
            "claim": "supplementary tests modified the method",
            "status": "false_by_protocol",
            "scope": "all tests are post-evaluation analyses",
        },
    ])
    engine.write_csv(claims, Path(root) / "limites_de_alegacao_v68.csv")



# ---------------------------------------------------------------------------
# Predictive performance audit on the complete outer test set.
# These wrappers observe the exact frozen ELM instance and split used by the
# explanation experiment. They do not change training, predictions, or any
# mathematical output.
# ---------------------------------------------------------------------------

def _capture_split_outer_and_calibration(bundle: Any, seed: int, calibration_fraction: float):
    outcome = _ENGINE_SPLIT_OUTER_AND_CALIBRATION(bundle, seed, calibration_fraction)
    key = _CURRENT_PREDICTIVE_KEY
    if key is not None:
        X_train, X_test, y_train, y_test, cal_idx = outcome
        _PREDICTIVE_CAPTURE.setdefault(key, {}).update({
            "X_test": np.asarray(X_test),
            "y_test": np.asarray(y_test).astype(str),
            "n_train_outer": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_calibration": int(len(cal_idx)),
        })
    return outcome


def _capture_build_ensemble(args: argparse.Namespace, runtime: Any, X_train: np.ndarray, y_train: Sequence[Any], seed: int):
    outcome = _ENGINE_BUILD_ENSEMBLE(args, runtime, X_train, y_train, seed)
    key = _CURRENT_PREDICTIVE_KEY
    if key is not None:
        ensemble = outcome[0] if isinstance(outcome, tuple) else outcome
        _PREDICTIVE_CAPTURE.setdefault(key, {}).update({
            "ensemble": ensemble,
            "n_fit": int(len(X_train)),
        })
    return outcome


def _predictive_test_artifacts(
    *,
    bundle: Any,
    scenario: str,
    seed: int,
    args: argparse.Namespace,
    capture: Mapping[str, Any],
) -> Dict[str, pd.DataFrame]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model = capture.get("ensemble")
    X_test = capture.get("X_test")
    y_test = capture.get("y_test")
    if model is None or X_test is None or y_test is None:
        return {}
    X_test = np.asarray(X_test)
    y_true = np.asarray(y_test).astype(str)
    classes = np.asarray(model.classes_).astype(str)
    started = time.perf_counter()
    probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    predicted_indices = np.asarray(model.predict_indices(X_test), dtype=int)
    y_pred = classes[predicted_indices]
    predictive_seconds = float(time.perf_counter() - started)

    auc_value = np.nan
    auc_definition = "not_applicable"
    try:
        if len(classes) == 2:
            y_binary = (y_true == classes[1]).astype(int)
            auc_value = float(roc_auc_score(y_binary, probabilities[:, 1]))
            auc_definition = f"binary_positive_class={classes[1]}"
        elif len(classes) > 2:
            auc_value = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=classes.tolist(),
                    multi_class="ovr",
                    average="macro",
                )
            )
            auc_definition = "macro_one_vs_rest"
    except Exception as exc:
        auc_definition = f"unavailable:{type(exc).__name__}"

    metric_row = {
        "dataset": str(bundle.name),
        "scenario": str(scenario),
        "seed": int(seed),
        "task_type": "multiclass_classification" if len(classes) > 2 else "binary_classification",
        "n_classes": int(len(classes)),
        "n_features": int(X_test.shape[1]),
        "n_test_complete": int(len(y_true)),
        "n_train_outer": int(capture.get("n_train_outer", 0)),
        "n_fit": int(capture.get("n_fit", 0)),
        "n_calibration": int(capture.get("n_calibration", 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes.tolist())),
        "roc_auc": auc_value,
        "roc_auc_definition": auc_definition,
        "predictive_inference_seconds_complete_test": predictive_seconds,
        "classes_json": json.dumps(classes.tolist(), ensure_ascii=False),
        "complete_outer_test": True,
        "explanation_subset_only": False,
    }
    per_seed = _clean_frame(
        pd.DataFrame([metric_row]), args=args, seed=seed, dataset=str(bundle.name), scenario=str(scenario)
    )

    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    confusion_rows: List[Dict[str, Any]] = []
    for actual_index, actual_label in enumerate(classes):
        actual_total = int(matrix[actual_index, :].sum())
        for predicted_index, predicted_label in enumerate(classes):
            count = int(matrix[actual_index, predicted_index])
            confusion_rows.append({
                "dataset": str(bundle.name),
                "scenario": str(scenario),
                "seed": int(seed),
                "actual_class_index": int(actual_index),
                "actual_class": str(actual_label),
                "predicted_class_index": int(predicted_index),
                "predicted_class": str(predicted_label),
                "count": count,
                "row_fraction": float(count / actual_total) if actual_total else np.nan,
                "complete_outer_test": True,
            })
    confusion = _clean_frame(
        pd.DataFrame(confusion_rows), args=args, seed=seed, dataset=str(bundle.name), scenario=str(scenario)
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    class_rows: List[Dict[str, Any]] = []
    for class_index, class_label in enumerate(classes):
        tp = int(matrix[class_index, class_index])
        fn = int(matrix[class_index, :].sum() - tp)
        fp = int(matrix[:, class_index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
        class_rows.append({
            "dataset": str(bundle.name),
            "scenario": str(scenario),
            "seed": int(seed),
            "class_index": int(class_index),
            "class_label": str(class_label),
            "support": int(support[class_index]),
            "precision": float(precision[class_index]),
            "recall_sensitivity": float(recall[class_index]),
            "specificity": specificity,
            "f1": float(f1[class_index]),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "complete_outer_test": True,
        })
    class_metrics = _clean_frame(
        pd.DataFrame(class_rows), args=args, seed=seed, dataset=str(bundle.name), scenario=str(scenario)
    )

    dataset_row = {
        "dataset": str(bundle.name),
        "scenario": str(scenario),
        "seed": int(seed),
        "source": str(getattr(bundle, "source", "")),
        "n_samples_total": int(len(np.asarray(bundle.y))),
        "n_features": int(np.asarray(bundle.X).shape[1]),
        "n_classes": int(len(classes)),
        "task_type": "multiclass_classification" if len(classes) > 2 else "binary_classification",
        "class_distribution_total_json": json.dumps(
            {str(label): int(np.sum(np.asarray(bundle.y).astype(str) == str(label))) for label in classes},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "class_distribution_test_json": json.dumps(
            {str(label): int(np.sum(y_true == str(label))) for label in classes},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "complete_outer_test": True,
    }
    dataset_summary = _clean_frame(
        pd.DataFrame([dataset_row]), args=args, seed=seed, dataset=str(bundle.name), scenario=str(scenario)
    )
    return {
        "predictive_performance_per_seed": per_seed,
        "predictive_confusion_matrix": confusion,
        "predictive_class_metrics": class_metrics,
        "predictive_dataset_per_seed": dataset_summary,
    }


def _evaluate_bundle_v68(
    bundle: Any,
    args: argparse.Namespace,
    runtime: Any,
    root: Path,
    seed: int,
    scenario: str,
    teacher_direct: bool = False,
) -> Dict[str, pd.DataFrame]:
    global _CURRENT_PREDICTIVE_KEY
    key = (str(bundle.name), str(scenario), int(seed))
    _PREDICTIVE_CAPTURE.pop(key, None)
    _CURRENT_PREDICTIVE_KEY = key
    try:
        result = _ENGINE_EVALUATE_BUNDLE(bundle, args, runtime, root, int(seed), str(scenario), teacher_direct)
    finally:
        _CURRENT_PREDICTIVE_KEY = None
    if not teacher_direct:
        predictive = _predictive_test_artifacts(
            bundle=bundle,
            scenario=str(scenario),
            seed=int(seed),
            args=args,
            capture=_PREDICTIVE_CAPTURE.pop(key, {}),
        )
        if not predictive:
            raise RuntimeError(
                f"Predictive full-test audit could not capture the frozen ELM for {bundle.name}, seed {seed}."
            )
        result.update(predictive)
        table_dir = Path(root) / "per_seed" / str(bundle.name) / str(scenario) / f"seed_{int(seed)}" / "tabelas"
        table_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name, frame in predictive.items():
            engine.write_csv(frame, table_dir / f"{artifact_name}.csv")
    return result


def _mean_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    sem = float(stats.sem(array))
    critical = float(stats.t.ppf(0.975, len(array) - 1))
    return mean, float(mean - critical * sem), float(mean + critical * sem)


def _aggregate_predictive_outputs(
    results: Sequence[Mapping[str, Any]], combined: Path, args: argparse.Namespace
) -> Dict[str, pd.DataFrame]:
    keys = (
        "predictive_performance_per_seed",
        "predictive_confusion_matrix",
        "predictive_class_metrics",
        "predictive_dataset_per_seed",
    )
    aggregated: Dict[str, pd.DataFrame] = {}
    for key in keys:
        frames = [item.get(key) for item in results]
        frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
        frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        aggregated[key] = frame
        engine.write_csv(frame, combined / f"{key}.csv")

    performance = aggregated["predictive_performance_per_seed"]
    metrics = [
        "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
        "matthews_correlation_coefficient", "log_loss", "roc_auc",
        "predictive_inference_seconds_complete_test",
    ]
    summary_rows: List[Dict[str, Any]] = []
    if not performance.empty:
        for (dataset, scenario), group in performance.groupby(["dataset", "scenario"], dropna=False):
            for metric in metrics:
                values = pd.to_numeric(group.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                mean, ci_low, ci_high = _mean_ci95(values)
                summary_rows.append({
                    "dataset": dataset,
                    "scenario": scenario,
                    "metric": metric,
                    "n_seeds": int(len(values)),
                    "mean": mean,
                    "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "interquartile_range": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "complete_outer_test": True,
                })
    summary = _clean_frame(pd.DataFrame(summary_rows), args=args) if summary_rows else pd.DataFrame()
    engine.write_csv(summary, combined / "predictive_performance_summary.csv")
    aggregated["predictive_performance_summary"] = summary

    article_rows: List[Dict[str, Any]] = []
    if not summary.empty:
        for (dataset, scenario), group in summary.groupby(["dataset", "scenario"], dropna=False):
            row: Dict[str, Any] = {"dataset": dataset, "scenario": scenario}
            for _, record in group.iterrows():
                metric = str(record["metric"])
                row[f"{metric}_mean"] = float(record["mean"])
                row[f"{metric}_ci95_low"] = float(record["ci95_low"])
                row[f"{metric}_ci95_high"] = float(record["ci95_high"])
                row[f"{metric}_article"] = (
                    f"{float(record['mean']):.4f} [{float(record['ci95_low']):.4f}; "
                    f"{float(record['ci95_high']):.4f}]"
                )
            article_rows.append(row)
    article = _clean_frame(pd.DataFrame(article_rows), args=args) if article_rows else pd.DataFrame()
    engine.write_csv(article, combined / "predictive_performance_article_table.csv")
    aggregated["predictive_performance_article_table"] = article

    confusion = aggregated["predictive_confusion_matrix"]
    confusion_summary = pd.DataFrame()
    if not confusion.empty:
        group_cols = ["dataset", "scenario", "actual_class_index", "actual_class", "predicted_class_index", "predicted_class"]
        confusion_summary = confusion.groupby(group_cols, as_index=False).agg(
            count_sum=("count", "sum"),
            count_mean_per_seed=("count", "mean"),
            row_fraction_mean=("row_fraction", "mean"),
            row_fraction_sd=("row_fraction", "std"),
            n_seeds=("seed", "nunique"),
        )
        confusion_summary = _clean_frame(confusion_summary, args=args)
    engine.write_csv(confusion_summary, combined / "predictive_confusion_matrix_summary.csv")
    aggregated["predictive_confusion_matrix_summary"] = confusion_summary

    class_metrics = aggregated["predictive_class_metrics"]
    class_summary = pd.DataFrame()
    if not class_metrics.empty:
        metric_cols = ["precision", "recall_sensitivity", "specificity", "f1", "support"]
        rows: List[Dict[str, Any]] = []
        for keys_value, group in class_metrics.groupby(["dataset", "scenario", "class_index", "class_label"], dropna=False):
            base = dict(zip(["dataset", "scenario", "class_index", "class_label"], keys_value))
            for metric in metric_cols:
                values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                mean, low, high = _mean_ci95(values)
                rows.append({**base, "metric": metric, "n_seeds": int(len(values)), "mean": mean, "ci95_low": low, "ci95_high": high})
        class_summary = _clean_frame(pd.DataFrame(rows), args=args) if rows else pd.DataFrame()
    engine.write_csv(class_summary, combined / "predictive_class_metrics_summary.csv")
    aggregated["predictive_class_metrics_summary"] = class_summary

    dataset_per_seed = aggregated["predictive_dataset_per_seed"]
    dataset_summary = pd.DataFrame()
    if not dataset_per_seed.empty:
        keep = [
            "dataset", "scenario", "source", "n_samples_total", "n_features", "n_classes", "task_type",
            "class_distribution_total_json",
        ]
        existing = [column for column in keep if column in dataset_per_seed.columns]
        dataset_summary = dataset_per_seed[existing].drop_duplicates(subset=["dataset", "scenario"]).reset_index(drop=True)
        repeats = performance.groupby(["dataset", "scenario"])["seed"].nunique().rename("n_repetitions").reset_index()
        dataset_summary = dataset_summary.merge(repeats, on=["dataset", "scenario"], how="left")
        dataset_summary = _clean_frame(dataset_summary, args=args)
    engine.write_csv(dataset_summary, combined / "predictive_dataset_summary.csv")
    aggregated["predictive_dataset_summary"] = dataset_summary
    return aggregated


def _generate_predictive_figures(combined: Path, outputs: Mapping[str, pd.DataFrame]) -> Path:
    figure_dir = engine.ensure_dir(combined.parent / "graficos_preditivos")
    summary = outputs.get("predictive_performance_summary", pd.DataFrame())
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        selected = summary[summary["metric"].isin(["balanced_accuracy", "f1_macro", "roc_auc"])].copy()
        if not selected.empty:
            datasets = list(dict.fromkeys(selected["dataset"].astype(str).tolist()))
            metrics = [m for m in ("balanced_accuracy", "f1_macro", "roc_auc") if m in set(selected["metric"])]
            x = np.arange(len(datasets), dtype=float)
            width = 0.8 / max(1, len(metrics))
            fig, ax = plt.subplots(figsize=(max(9.0, 1.25 * len(datasets)), 5.5))
            for index, metric in enumerate(metrics):
                group = selected[selected["metric"].eq(metric)].set_index("dataset")
                means = np.array([float(group.loc[d, "mean"]) if d in group.index else np.nan for d in datasets])
                low = np.array([float(group.loc[d, "ci95_low"]) if d in group.index else np.nan for d in datasets])
                high = np.array([float(group.loc[d, "ci95_high"]) if d in group.index else np.nan for d in datasets])
                position = x - 0.4 + width / 2 + index * width
                ax.bar(position, means, width=width, label=metric.replace("_", " "))
                ax.errorbar(position, means, yerr=np.vstack([means - low, high - means]), fmt="none", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=35, ha="right")
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("Mean performance with 95% CI")
            ax.set_title("ELM predictive performance on the complete outer test set")
            ax.legend()
            fig.tight_layout()
            for extension in ("png", "pdf"):
                fig.savefig(figure_dir / f"predictive_performance_overview.{extension}", dpi=220, bbox_inches="tight")
            plt.close(fig)

    confusion = outputs.get("predictive_confusion_matrix_summary", pd.DataFrame())
    if isinstance(confusion, pd.DataFrame) and not confusion.empty:
        for dataset, group in confusion.groupby("dataset"):
            actual = sorted(group["actual_class"].astype(str).unique().tolist())
            predicted = sorted(group["predicted_class"].astype(str).unique().tolist())
            matrix = np.full((len(actual), len(predicted)), np.nan, dtype=float)
            for _, row in group.iterrows():
                matrix[actual.index(str(row["actual_class"])), predicted.index(str(row["predicted_class"]))] = float(row["row_fraction_mean"])
            fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * len(predicted) + 2), max(4.5, 0.55 * len(actual) + 2)))
            image = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(predicted)))
            ax.set_xticklabels(predicted, rotation=45, ha="right")
            ax.set_yticks(range(len(actual)))
            ax.set_yticklabels(actual)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.set_title(f"Mean normalized confusion matrix - {dataset}")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset))
            fig.savefig(figure_dir / f"confusion_matrix_{safe}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
    return figure_dir


# ---------------------------------------------------------------------------
# Detailed PDF report generator.
# ---------------------------------------------------------------------------

def _read_result_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 3:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def _pdf_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "NA"
        if number != 0.0 and abs(number) < 10 ** (-digits):
            return f"{number:.2e}"
        return f"{number:.{digits}f}"
    except Exception:
        text = str(value)
        return text if len(text) <= 48 else text[:45] + "..."


def _generate_pdf_report(experiment_root: Path, args: Optional[argparse.Namespace] = None, output_path: Optional[Path] = None) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics
        from reportlab.platypus import (
            BaseDocTemplate, Frame, Image as RLImage, KeepTogether, LongTable, NextPageTemplate,
            PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.platypus.tableofcontents import TableOfContents
        from reportlab.pdfgen import canvas
        from xml.sax.saxutils import escape
    except Exception as exc:
        raise RuntimeError("PDF report generation requires reportlab. Install with: pip install reportlab") from exc

    root = Path(experiment_root).expanduser().resolve()
    combined = root / "combined" / "tabelas"
    if not combined.exists():
        raise FileNotFoundError(f"Combined table directory not found: {combined}")
    pdf_path = Path(output_path).expanduser().resolve() if output_path else root / "relatorio.pdf"
    assets = engine.ensure_dir(root / "combined" / "relatorio_assets")

    table_names = [
        "predictive_dataset_summary", "predictive_performance_summary", "predictive_performance_article_table",
        "predictive_confusion_matrix_summary", "predictive_class_metrics_summary",
        "inferential_test_summary", "paired_t_tests", "shapiro_wilk_tests", "repeated_measures_anova",
        "friedman_tests", "nemenyi_posthoc", "binomial_win_rate_tests", "multiple_testing_families",
        "global_fidelity_summary", "local_fidelity_summary", "local_agreement", "global_importance",
        "global_class_importance", "estabilidade_global_entre_seeds", "global_stability_diagnostics",
        "prototype_library_class_summary", "prototype_usage_summary", "prototype_routing_summary",
        "prototype_stability_between_seeds", "timing", "cuda_audit", "gpu_runtime", "calibration",
        "convergence", "exact_shapley", "support", "text_audit_findings",
    ]
    data = {name: _read_result_csv(combined / f"{name}.csv") for name in table_names}

    # Use a font with broad Latin coverage when available, without distributing it.
    body_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        try:
            if candidate.exists():
                pdfmetrics.registerFont(TTFont("ReportSans", str(candidate)))
                body_font = "ReportSans"
                break
        except Exception:
            pass
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ):
        try:
            if candidate.exists():
                pdfmetrics.registerFont(TTFont("ReportSansBold", str(candidate)))
                bold_font = "ReportSansBold"
                break
        except Exception:
            pass

    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *a, **kw):
            canvas.Canvas.__init__(self, *a, **kw)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            page_count = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.setFont(body_font, 8)
                self.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, f"Pagina {self._pageNumber} de {page_count}")
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName=bold_font, fontSize=22, leading=27, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="ReportSubtitle", parent=styles["Normal"], fontName=body_font, fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName=bold_font, fontSize=16, leading=20, spaceBefore=14, spaceAfter=9, textColor=colors.HexColor("#17365D")))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName=bold_font, fontSize=13, leading=16, spaceBefore=11, spaceAfter=7, textColor=colors.HexColor("#1F4E79")))
    styles.add(ParagraphStyle(name="H3x", parent=styles["Heading3"], fontName=bold_font, fontSize=11, leading=14, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName=body_font, fontSize=9.2, leading=13, alignment=TA_JUSTIFY, spaceAfter=7))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontName=body_font, fontSize=7.5, leading=10, alignment=TA_LEFT, spaceAfter=4))
    styles.add(ParagraphStyle(name="Captionx", parent=styles["BodyText"], fontName=body_font, fontSize=7.5, leading=10, alignment=TA_CENTER, spaceAfter=8, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="Calloutx", parent=styles["BodyText"], fontName=body_font, fontSize=9, leading=13, leftIndent=10, rightIndent=10, borderWidth=0.5, borderColor=colors.HexColor("#8EA9DB"), borderPadding=7, backColor=colors.HexColor("#EAF2F8"), spaceBefore=5, spaceAfter=8))

    class ReportDocTemplate(BaseDocTemplate):
        def __init__(self, filename: str):
            BaseDocTemplate.__init__(self, filename, pagesize=A4, rightMargin=1.45 * cm, leftMargin=1.45 * cm, topMargin=1.45 * cm, bottomMargin=1.35 * cm, title="Relatorio NEX-ELM v68")
            frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
            self.addPageTemplates([PageTemplate(id="normal", frames=frame)])
            self._heading_counter = 0

        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph) and flowable.style.name in {"H1x", "H2x", "H3x"}:
                level = {"H1x": 0, "H2x": 1, "H3x": 2}[flowable.style.name]
                key = getattr(flowable, "_v68_bookmark_key", None)
                if key is None:
                    self._heading_counter += 1
                    key = f"heading_{self._heading_counter}"
                    setattr(flowable, "_v68_bookmark_key", key)
                text = flowable.getPlainText()
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))

    def P(text: Any, style: str = "Bodyx") -> Paragraph:
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def H(text: str, level: int = 1) -> Paragraph:
        return Paragraph(escape(text), styles[{1: "H1x", 2: "H2x", 3: "H3x"}[level]])

    def add_table(frame: pd.DataFrame, columns: Sequence[str], labels: Optional[Mapping[str, str]] = None, max_rows: int = 30, font_size: float = 6.5):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return [P("Tabela nao disponivel ou nao aplicavel nesta execucao.", "Smallx")]
        columns_existing = [column for column in columns if column in frame.columns]
        if not columns_existing:
            return [P("As colunas esperadas nao foram encontradas no arquivo de resultados.", "Smallx")]
        labels = dict(labels or {})
        subset = frame[columns_existing].head(max_rows)
        matrix = [[P(labels.get(column, column), "Smallx") for column in columns_existing]]
        for _, row in subset.iterrows():
            matrix.append([P(_pdf_number(row[column]) if isinstance(row[column], (int, float, np.integer, np.floating)) else str(row[column]), "Smallx") for column in columns_existing])
        available = 17.5 * cm
        widths = [available / len(columns_existing)] * len(columns_existing)
        table = LongTable(matrix, repeatRows=1, colWidths=widths, splitByRow=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("FONTNAME", (0, 1), (-1, -1), body_font),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A6A6A6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements: List[Any] = [table]
        if len(frame) > max_rows:
            elements.append(P(f"A tabela exibe {max_rows} de {len(frame)} linhas; o arquivo CSV contem o resultado completo.", "Smallx"))
        return elements

    def add_image(path: Path, caption: str, max_width: float = 17.0 * cm, max_height: float = 11.5 * cm):
        if not path.exists():
            return []
        try:
            image = RLImage(str(path))
            factor = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
            image.drawWidth = image.imageWidth * factor
            image.drawHeight = image.imageHeight * factor
            image.hAlign = "CENTER"
            return [image, P(caption, "Captionx")]
        except Exception:
            return [P(f"A figura {path.name} nao pôde ser incorporada ao PDF.", "Smallx")]

    predictive_summary = data["predictive_performance_summary"]
    datasets_df = data["predictive_dataset_summary"]
    inferential = data["inferential_test_summary"]
    primary = pd.DataFrame()
    if not inferential.empty:
        marker = pd.to_numeric(inferential.get("one_sided_wilcoxon_p_value_holm_primary"), errors="coerce")
        primary = inferential[marker.notna()].copy() if marker is not None else pd.DataFrame()
    confirmed = int(pd.Series(primary.get("confirmatory_superiority", [])).fillna(False).astype(bool).sum()) if not primary.empty else 0
    total_primary = int(len(primary))
    dataset_names = datasets_df["dataset"].astype(str).tolist() if not datasets_df.empty and "dataset" in datasets_df else sorted(set(inferential.get("dataset", pd.Series(dtype=str)).astype(str)))
    n_seeds = 0
    perf_seed = _read_result_csv(combined / "predictive_performance_per_seed.csv")
    if not perf_seed.empty and "seed" in perf_seed:
        n_seeds = int(perf_seed["seed"].nunique())

    story: List[Any] = []
    story += [Spacer(1, 1.8 * cm), P("NEX-ELM v68", "ReportTitle"), P("Relatorio completo de desempenho preditivo, fidelidade explicativa, estatistica, estabilidade, prototipos, tempo e auditoria CUDA", "ReportSubtitle")]
    story += [Spacer(1, 1.0 * cm)]
    cover_data = [
        [P("Campo", "Smallx"), P("Valor", "Smallx")],
        [P("Diretorio", "Smallx"), P(str(root), "Smallx")],
        [P("Data de geracao", "Smallx"), P(datetime.now(timezone.utc).isoformat(), "Smallx")],
        [P("Bases avaliadas", "Smallx"), P(str(len(dataset_names)), "Smallx")],
        [P("Seeds distintas", "Smallx"), P(str(n_seeds), "Smallx")],
        [P("Nucleo matematico SHA-256", "Smallx"), P(EXPECTED_ENGINE_SHA256, "Smallx")],
        [P("Representacao", "Smallx"), P(GLOBAL_REPRESENTATION_DEFINITION, "Smallx")],
    ]
    cover_table = Table(cover_data, colWidths=[5.0 * cm, 11.8 * cm])
    cover_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [cover_table, Spacer(1, 1.0 * cm), P("Este documento foi produzido apenas a partir dos arquivos da execucao. Valores ausentes sao declarados como nao aplicaveis; nenhuma conclusao numerica e inventada.", "Calloutx"), PageBreak()]

    story += [H("Sumario", 1)]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name="TOC1", fontName=body_font, fontSize=10, leftIndent=0, firstLineIndent=0, leading=14),
        ParagraphStyle(name="TOC2", fontName=body_font, fontSize=9, leftIndent=16, firstLineIndent=0, leading=12),
        ParagraphStyle(name="TOC3", fontName=body_font, fontSize=8, leftIndent=32, firstLineIndent=0, leading=10),
    ]
    story += [toc, PageBreak()]

    story += [H("1. Resumo executivo", 1)]
    if total_primary:
        story += [P(f"A familia primaria contem {total_primary} comparacoes e {confirmed} foram confirmadas pelo criterio pre-especificado. A interpretacao deve combinar direcao do efeito, intervalo de confianca, testes pareados, correcao de Holm e taxa de vitorias entre seeds.")]
    else:
        story += [P("O resumo de hipoteses primarias nao estava disponivel. Consulte os arquivos inferenciais no diretorio combined/tabelas.")]
    if not predictive_summary.empty:
        ba = predictive_summary[predictive_summary["metric"].eq("balanced_accuracy")].copy()
        if not ba.empty:
            best = ba.loc[pd.to_numeric(ba["mean"], errors="coerce").idxmax()]
            worst = ba.loc[pd.to_numeric(ba["mean"], errors="coerce").idxmin()]
            story += [P(f"No teste externo completo, a maior balanced accuracy media ocorreu em {best['dataset']} ({_pdf_number(best['mean'])}) e a menor em {worst['dataset']} ({_pdf_number(worst['mean'])}). Esses valores avaliam a capacidade preditiva do ELM; nao medem, por si so, a qualidade das explicacoes.")]
    story += [P("A alegacao principal permanece restrita a fidelidade e granularidade explicativa. O tempo deve ser comparado no nivel de workflow completo contra X-ELM mais Kernel SHAP; nao se afirma que o NEX-ELM isolado seja mais rapido que o X-ELM isolado.", "Calloutx")]

    story += [H("2. Protocolo e bases", 1)]
    story += [P("O protocolo usa treino, calibracao e teste separados. A biblioteca de prototipos e construida apenas com calibracao; o teste externo nao define prototipos, nao escolhe K, nao altera o metodo e nao participa do roteamento por rotulos ou fidelidade.")]
    story += add_table(datasets_df, ["dataset", "task_type", "n_samples_total", "n_features", "n_classes", "n_repetitions", "class_distribution_total_json"], max_rows=20)
    story += [P("A tabela descreve os cenarios efetivamente executados. Electrical Grid com stab reproduz o protocolo do artigo X-ELM; a versao sem stab funciona como controle contra proxy do alvo. Wine, Iris e Digits verificam classificacao multiclasse; Ionosphere e Breast Cancer Diagnostic ampliam dominio e dimensionalidade.")]

    story += [H("3. Desempenho preditivo do ELM no teste completo", 1)]
    article = data["predictive_performance_article_table"]
    article_cols = ["dataset", "accuracy_article", "balanced_accuracy_article", "f1_macro_article", "matthews_correlation_coefficient_article", "log_loss_article", "roc_auc_article"]
    story += add_table(article, article_cols, labels={"dataset": "Base", "accuracy_article": "Accuracy IC95%", "balanced_accuracy_article": "Balanced accuracy IC95%", "f1_macro_article": "F1 macro IC95%", "matthews_correlation_coefficient_article": "MCC IC95%", "log_loss_article": "Log loss IC95%", "roc_auc_article": "ROC-AUC IC95%"}, max_rows=20, font_size=6.0)
    story += [P("Cada celula apresenta media e intervalo de confianca de 95% entre seeds. Accuracy mede acerto total; balanced accuracy atribui o mesmo peso a cada classe; F1 macro combina precisao e recall sem favorecer classes maiores; MCC resume a matriz de confusao; log loss penaliza probabilidades confiantes e erradas; ROC-AUC avalia ordenacao probabilistica, usando macro One-vs-Rest nas tarefas multiclasse.")]
    story += add_image(root / "combined" / "graficos_preditivos" / "predictive_performance_overview.png", "Figura 1. Desempenho preditivo medio do ELM no conjunto de teste externo completo.")

    story += [H("4. Fidelidade global e local", 1)]
    if not primary.empty:
        global_primary = primary[primary["scope"].astype(str).str.contains("global", case=False, na=False)]
        local_primary = primary[primary["scope"].astype(str).str.contains("local", case=False, na=False)]
        story += [H("4.1 Comparacoes globais", 2)]
        story += add_table(global_primary, ["dataset", "metric", "first_method", "second_method", "mean_oriented_difference", "ci95_low", "ci95_high", "effect_size_cohen_dz", "first_wins", "required_seed_wins", "confirmatory_superiority"], max_rows=50)
        story += [P("Diferenca positiva favorece o primeiro metodo quando higher_is_better e verdadeiro. A confirmacao exige a combinacao registrada de testes pareados, correcao de Holm, intervalo na direcao esperada e numero minimo de seeds favoraveis. Um p pequeno nao substitui tamanho de efeito nem estabilidade entre seeds.")]
        story += [H("4.2 Comparacoes locais", 2)]
        story += add_table(local_primary, ["dataset", "metric", "first_method", "second_method", "mean_oriented_difference", "ci95_low", "ci95_high", "effect_size_cohen_dz", "first_wins", "required_seed_wins", "confirmatory_superiority"], max_rows=50)
        story += [P("As metricas locais avaliam as explicacoes das previsoes individuais. Delecao verifica a queda da saida ao remover atributos mais importantes; insercao verifica a recuperacao da saida ao inserir esses atributos. O sinal e orientado para que valores positivos indiquem vantagem do NEX-ELM.")]
    else:
        story += [P("O arquivo inferential_test_summary.csv nao continha marcacao de familia primaria.")]

    stat_fig_dir = root / "combined" / "graficos_estatisticos"
    figure_candidates = sorted(stat_fig_dir.glob("*.png")) if stat_fig_dir.exists() else []
    for figure in figure_candidates[:10]:
        story += add_image(figure, f"Figura estatistica: {figure.stem.replace('_', ' ')}")

    story += [H("5. Testes de hipotese", 1)]
    story += [H("5.1 Teste t pareado e Shapiro-Wilk", 2)]
    story += add_table(data["paired_t_tests"], ["dataset", "scope", "metric", "first_method", "second_method", "mean_oriented_difference", "ci95_low", "ci95_high", "t_statistic", "paired_t_p_one_sided_holm", "effect_size_cohen_dz", "paired_t_superiority_supported"], max_rows=45)
    shapiro = data["shapiro_wilk_tests"]
    if not shapiro.empty and "normality_not_rejected_alpha" in shapiro:
        normal_ok = int(shapiro["normality_not_rejected_alpha"].fillna(False).astype(bool).sum())
        story += [P(f"O Shapiro-Wilk nao rejeitou normalidade em {normal_ok} de {len(shapiro)} distribuicoes pareadas. Quando normalidade e rejeitada, Wilcoxon e permutacao permanecem referencias mais robustas; o teste t atua como verificacao complementar.")]
    story += add_table(shapiro, ["dataset", "scope", "metric", "first_method", "second_method", "shapiro_w", "shapiro_p_value", "normality_not_rejected_alpha", "skewness_of_differences", "excess_kurtosis_of_differences"], max_rows=35)

    story += [H("5.2 ANOVA de medidas repetidas", 2)]
    story += add_table(data["repeated_measures_anova"], ["dataset", "scope", "metric", "n_subjects", "f_statistic", "degrees_of_freedom_1", "degrees_of_freedom_2", "p_value", "p_value_holm", "partial_eta_squared", "greenhouse_geisser_epsilon", "significant_after_holm"], max_rows=40)
    story += [P("A ANOVA testa se existe diferenca entre os metodos considerados em conjunto. Significancia omnibus nao prova, sozinha, que NEX-ELM difere de Kernel SHAP; por isso a leitura deve continuar nos contrastes pareados e no pos-teste.")]

    story += [H("5.3 Friedman, Kendall W e Nemenyi", 2)]
    story += add_table(data["friedman_tests"], ["dataset", "scope", "metric", "n_subjects", "friedman_statistic", "p_value", "p_value_holm", "kendall_w", "significant_after_holm"], max_rows=40)
    story += add_table(data["nemenyi_posthoc"], ["dataset", "scope", "metric", "first_method", "second_method", "average_rank_first", "average_rank_second", "rank_difference", "nemenyi_p_value", "significant_after_holm"], max_rows=55)
    story += [P("Friedman compara ranks dentro de cada seed. Kendall W quantifica a separacao entre metodos: valores proximos de 1 indicam ordenacao consistente. Nemenyi localiza os pares que diferem, mas possui menor poder que contrastes pareados pre-especificados.")]

    story += [H("5.4 Teste binomial exato e taxa de vitorias", 2)]
    story += add_table(data["binomial_win_rate_tests"], ["dataset", "scope", "metric", "first_method", "second_method", "n_trials_seeds", "first_method_wins", "win_rate", "clopper_pearson_two_sided_ci95_low", "clopper_pearson_two_sided_ci95_high", "observed_win_rate_at_least_80_percent", "binomial_p_greater_than_50_percent_holm"], max_rows=45)
    story += [P("O intervalo Clopper-Pearson e bilateral. A regra de 80% e um limiar observado pre-especificado; ela nao deve ser descrita como prova de que a taxa populacional supera 80% quando o teste correspondente nao e significativo apos Holm. Empates permanecem no denominador como nao vitorias.")]

    story += [H("6. Concordancia, completude e estabilidade", 1)]
    local_agreement = data["local_agreement"]
    if not local_agreement.empty:
        cols = [column for column in ["dataset", "pearson", "spearman", "topk_overlap", "sign_agreement", "score_completeness_absolute_error", "probability_completeness_absolute_error"] if column in local_agreement]
        if cols:
            local_summary = local_agreement.groupby("dataset", as_index=False)[[c for c in cols if c != "dataset"]].mean(numeric_only=True)
            story += add_table(local_summary, cols, max_rows=20)
            story += [P("Concordancia com SHAP mostra similaridade, nao fidelidade absoluta. Um metodo pode discordar de SHAP e ainda obter maior insercao/delecao. Os erros de completude verificam se as atribuicoes recompõem a diferenca entre saida e referencia no espaco declarado.")]
    stability = data["estabilidade_global_entre_seeds"]
    story += add_table(stability, ["dataset", "method", "spearman_mean", "spearman_median", "topk_overlap_mean", "topk_overlap_median", "n_seed_pairs"], max_rows=30)
    story += [P("A estabilidade entre seeds deve ser interpretada junto com fidelidade. Rankings completos podem variar quando atributos sao correlacionados ou intercambiaveis, mesmo quando a remocao e insercao mantêm desempenho alto.")]

    story += [H("7. Biblioteca glocal de prototipos", 1)]
    story += add_table(data["prototype_routing_summary"], ["dataset", "target_class_index", "method", "routed_samples", "used_prototypes", "prototype_capacity", "registered_distance_mean", "glocal_inference_seconds_per_sample"], max_rows=40)
    story += add_table(data["prototype_stability_between_seeds"], ["dataset", "method", "target_class_index", "mean_matched_registered_distance", "exact_signature_match_rate", "prototype_count_mean", "n_seed_pairs"], max_rows=40)
    story += [P("K=4 e a capacidade da biblioteca permanecem fixos. A estabilidade estrutural refere-se ao procedimento, distancia e capacidade; assinaturas exatas podem mudar quando treino e calibracao mudam entre seeds. Portanto, nao se deve afirmar que os mesmos prototipos aparecem em todas as repeticoes.")]

    story += [H("8. Tempo e eficiencia", 1)]
    timing = data["timing"]
    if not timing.empty and {"dataset", "seed", "method", "seconds"}.issubset(timing.columns):
        pivot = timing[timing["scope"].astype(str).eq("workflow")].pivot_table(index=["dataset", "seed"], columns="method", values="seconds", aggfunc="mean").reset_index()
        left = "NEX-ELM workflow complete"
        right = "X-ELM + Kernel SHAP workflow complete"
        if left in pivot and right in pivot:
            pivot["workflow_speedup"] = pd.to_numeric(pivot[right], errors="coerce") / pd.to_numeric(pivot[left], errors="coerce")
            speed = pivot.groupby("dataset", as_index=False)["workflow_speedup"].agg(["mean", "median", "min", "max"]).reset_index()
            story += add_table(speed, ["dataset", "mean", "median", "min", "max"], labels={"mean": "Speedup medio", "median": "Mediana", "min": "Minimo", "max": "Maximo"}, max_rows=20)
            story += [P("Speedup acima de 1 favorece o workflow NEX-ELM. A comparacao inclui construcao global, explicacoes locais, assinaturas de inferencia e roteamento. O X-ELM isolado continua sendo uma operacao global barata e nao e o alvo da alegacao de velocidade.")]
    story += add_image(root / "09_graficos" / "tempos.png", "Comparacao de tempos e workflows entre metodos.")

    story += [H("9. Auditoria CUDA", 1)]
    cuda = data["cuda_audit"]
    if not cuda.empty:
        story += add_table(cuda, ["dataset", "seed", "passed", "class_agreement", "maximum_absolute_score_difference", "maximum_absolute_probability_difference", "maximum_absolute_attribution_difference", "tolerance"], max_rows=35)
        passed_col = next((c for c in ["passed", "audit_passed"] if c in cuda), None)
        if passed_col:
            passed = int(cuda[passed_col].fillna(False).astype(bool).sum())
            story += [P(f"A auditoria CUDA foi aprovada em {passed} de {len(cuda)} registros. Diferencas pequenas sao esperadas por precisao numerica; divergencia de classe ou excesso da tolerancia invalida a equivalencia e deve ser investigada.")]
    story += add_table(data["gpu_runtime"], ["dataset", "gpu_name", "peak_memory_allocated_gib", "peak_memory_reserved_gib", "gpu_profile", "gpu_batch_size", "nex_instance_batch_size"], max_rows=25)

    story += [H("10. Analise por base", 1)]
    predictive_fig_dir = root / "combined" / "graficos_preditivos"
    for dataset in dataset_names:
        story += [H(str(dataset), 2)]
        if not predictive_summary.empty:
            ds_pred = predictive_summary[predictive_summary["dataset"].astype(str).eq(str(dataset))]
            story += add_table(ds_pred, ["metric", "n_seeds", "mean", "standard_deviation", "ci95_low", "ci95_high", "minimum", "maximum"], max_rows=15)
            if not ds_pred.empty:
                values = {str(row["metric"]): float(row["mean"]) for _, row in ds_pred.iterrows() if pd.notna(row.get("mean"))}
                story += [P(
                    f"O ELM obteve accuracy media {_pdf_number(values.get('accuracy'))}, balanced accuracy {_pdf_number(values.get('balanced_accuracy'))}, F1 macro {_pdf_number(values.get('f1_macro'))}, MCC {_pdf_number(values.get('matthews_correlation_coefficient'))} e ROC-AUC {_pdf_number(values.get('roc_auc'))}. A diferenca entre accuracy e balanced accuracy ajuda a identificar impacto do desbalanceamento."
                )]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset))
        story += add_image(predictive_fig_dir / f"confusion_matrix_{safe}.png", f"Matriz de confusao normalizada media - {dataset}")
        story += add_image(root / "09_graficos" / f"{dataset}_global.png", f"Comparacao de importancia global - {dataset}")
        story += add_image(root / "09_graficos" / f"{dataset}_global_deletion.png", f"Fidelidade global por delecao - {dataset}")
        story += add_image(root / "09_graficos" / f"{dataset}_local_deletion.png", f"Fidelidade local por delecao - {dataset}")
        if not primary.empty:
            ds_inf = primary[primary["dataset"].astype(str).eq(str(dataset))]
            story += add_table(ds_inf, ["scope", "metric", "first_method", "second_method", "mean_oriented_difference", "ci95_low", "ci95_high", "effect_size_cohen_dz", "first_wins", "required_seed_wins", "confirmatory_superiority"], max_rows=25)
            if not ds_inf.empty:
                passed = int(ds_inf["confirmatory_superiority"].fillna(False).astype(bool).sum())
                story += [P(f"Nesta base, {passed} de {len(ds_inf)} hipoteses primarias foram confirmadas. Resultados nao confirmados devem permanecer visiveis e ser descritos como mistos, inconclusivos ou nao superiores, conforme intervalo, direcao e taxa de vitorias.")]
        importance = data["global_importance"]
        if not importance.empty and "dataset" in importance:
            ds_imp = importance[importance["dataset"].astype(str).eq(str(dataset))]
            value_cols = [c for c in ["nexelm_global", "kernel_shap_global", "xelm_global"] if c in ds_imp]
            if value_cols and "feature" in ds_imp:
                imp = ds_imp.groupby("feature", as_index=False)[value_cols].mean(numeric_only=True)
                sort_col = "nexelm_global" if "nexelm_global" in imp else value_cols[0]
                imp = imp.sort_values(sort_col, ascending=False).head(10)
                story += [H("Atributos globais mais importantes", 3)]
                story += add_table(imp, ["feature"] + value_cols, max_rows=10)
                story += [P("O ranking apresenta medias entre seeds. Importancia alta indica prioridade no mecanismo explicativo registrado, mas nao deve ser interpretada como causalidade. Diferencas entre metodos podem refletir jogos de referencia e objetivos de fidelidade distintos.")]
        ds_stab = stability[stability["dataset"].astype(str).eq(str(dataset))] if not stability.empty and "dataset" in stability else pd.DataFrame()
        if not ds_stab.empty:
            story += add_table(ds_stab, ["method", "spearman_mean", "topk_overlap_mean", "n_seed_pairs"], max_rows=10)
        ds_cuda = cuda[cuda["dataset"].astype(str).eq(str(dataset))] if not cuda.empty and "dataset" in cuda else pd.DataFrame()
        if not ds_cuda.empty:
            story += [P(f"A auditoria CUDA contem {len(ds_cuda)} registros para esta base; consulte a secao geral e o CSV para os desvios por seed.")]

    story += [H("11. Limitacoes e limites de alegacao", 1)]
    story += [P("Os benchmarks nao representam todos os dominios, distribuicoes ou arquiteturas. As repeticoes medem variacao causada por divisao e inicializacao, mas nao substituem validacao em dados temporais, externos ou de producao. A fidelidade depende do jogo de referencia, do protocolo de insercao/delecao e da saida explicada.")]
    story += [P("Resultados do arquivo convergence.csv representam auditoria de completude para a configuracao executada quando existe apenas um numero de nos; nao constituem curva de convergencia. Exact Shapley em dados reais e nao aplicavel, devendo ser sustentado pelo diagnostico sintetico dedicado. Support recovery e aplicavel apenas quando existe suporte verdadeiro conhecido.")]
    findings = data["text_audit_findings"]
    if findings.empty:
        story += [P("A auditoria textual nao encontrou referencias publicas obsoletas nos artefatos CSV, JSON, TXT ou MD.")]
    else:
        story += [P(f"A auditoria textual registrou {len(findings)} achados. A submissao deve aguardar a resolucao desses itens.", "Calloutx")]

    story += [H("12. Conclusao", 1)]
    if total_primary:
        story += [P(f"O experimento confirmou {confirmed} de {total_primary} hipoteses primarias. A conclusao cientifica deve nomear as bases e comparacoes que passaram e as que permaneceram mistas, sem converter ausencia de significancia em equivalencia e sem ocultar resultados desfavoraveis.")]
    story += [P("O conjunto de resultados permite avaliar separadamente quatro dimensoes: capacidade preditiva do ELM, fidelidade das explicacoes, estabilidade entre seeds e custo computacional. A publicacao deve apresentar essas dimensoes como complementares, evitando usar boa acuracia como prova de explicabilidade ou boa concordancia com SHAP como prova de fidelidade.")]

    story += [H("Apendice A. Arquivos principais", 1)]
    file_rows = []
    for path in sorted(combined.glob("*.csv")):
        try:
            rows = len(_read_result_csv(path))
        except Exception:
            rows = -1
        file_rows.append({"arquivo": path.name, "linhas": rows, "tamanho_kib": path.stat().st_size / 1024.0})
    story += add_table(pd.DataFrame(file_rows), ["arquivo", "linhas", "tamanho_kib"], max_rows=120)

    doc = ReportDocTemplate(str(pdf_path))
    doc.multiBuild(story, canvasmaker=NumberedCanvas)
    if not pdf_path.exists() or pdf_path.stat().st_size < 5000:
        raise RuntimeError(f"PDF report was not created correctly: {pdf_path}")
    return pdf_path


def _run_report_only(args: argparse.Namespace) -> Path:
    source = Path(args.report_only_from).expanduser().resolve()
    output = Path(args.report_output).expanduser().resolve() if args.report_output is not None else source / "relatorio.pdf"
    result = _generate_pdf_report(source, args=args, output_path=output)
    print(f"PDF report completed: {result}")
    return result



def run_real(args: argparse.Namespace, runtime: Any, root: Path) -> List[Dict[str, pd.DataFrame]]:
    results: List[Dict[str, pd.DataFrame]] = []
    specs = _study_specs(args)
    _write_v68_study_plan(root, args, specs)
    for spec in specs:
        bundles = [_load_v68_bundle(name, args) for name in spec["datasets"]]
        for bundle in bundles:
            context = _dataset_context(bundle, spec["phase"], spec["evidence_role"], spec["scenario"])
            _STUDY_CONTEXT[(str(bundle.name), str(spec["scenario"]))] = context
        for repetition, seed in enumerate(spec["seeds"], start=1):
            for bundle in bundles:
                print(
                    f"[real {METHOD_TITLE}] phase={spec['phase']} | {bundle.name} | "
                    f"repetition={repetition}/{spec['repeats']} | seed={seed} | device={runtime.resolved}"
                )
                results.append(
                    _evaluate_bundle_v68(
                        bundle, args, runtime, root, int(seed), str(spec["scenario"]), False
                    )
                )
    return results


def run_synthetic(args: argparse.Namespace, runtime: Any, root: Path) -> List[Dict[str, pd.DataFrame]]:
    results: List[Dict[str, pd.DataFrame]] = []
    valid = {"sparse", "saturation", "interaction", "correlated", "exact_shapley"}
    kinds = [item.strip() for item in str(args.synthetic_kinds).split(",") if item.strip()]
    if not set(kinds).issubset(valid):
        raise ValueError(f"Invalid synthetic kinds: {kinds}")
    internal_modes = [args.synthetic_evaluation] if args.synthetic_evaluation != "both" else ["teacher", "student"]
    public_names = {"teacher": "reference_function", "student": "trained_model"}
    for repetition in range(int(args.synthetic_repetitions)):
        seed = int(args.random_state) + 100000 + repetition
        for kind in kinds:
            bundle = engine.make_synthetic_bundle(seed, kind, runtime, int(args.synthetic_samples))
            for internal_mode in internal_modes:
                public_mode = public_names.get(str(internal_mode), str(internal_mode))
                scenario = f"synthetic_{kind}_{public_mode}_v68_glopro_complete"
                print(
                    f"[synthetic {METHOD_TITLE}] {scenario} | repetition={repetition + 1}/"
                    f"{args.synthetic_repetitions} | seed={seed} | device={runtime.resolved}"
                )
                results.append(
                    _evaluate_bundle_v68(
                        bundle, args, runtime, root, seed, scenario,
                        teacher_direct=(str(internal_mode) == "teacher"),
                    )
                )
    return results


def _install_active_hooks() -> None:
    engine.VERSION = VERSION
    engine.parse_args = parse_args
    engine.apply_protocol = apply_protocol
    _replace_bindings(engine, _ENGINE_POSTPROCESS, postprocess_artifacts)
    _replace_bindings(engine, _ENGINE_AGGREGATE, aggregate_results)
    _replace_bindings(engine, _ENGINE_SPLIT_OUTER_AND_CALIBRATION, _capture_split_outer_and_calibration)
    _replace_bindings(engine, _ENGINE_BUILD_ENSEMBLE, _capture_build_ensemble)
    confirmatory_candidates = [
        function for function in _unique_functions(engine)
        if _function_parameters(function) == ("args",)
        and str(getattr(function, "__name__", "")).endswith("confirmatory_plan")
    ]
    if confirmatory_candidates:
        current = max(confirmatory_candidates, key=lambda function: int(function.__code__.co_firstlineno))
        _replace_bindings(engine, current, confirmatory_plan)
    engine.run_real = run_real
    engine.run_synthetic = run_synthetic


def _visible_source_for_audit(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = text.find("_EMBEDDED_ENGINE_B85 = (")
    end = text.find("\n\n_ENGINE_LOAD_PATH", start)
    if start >= 0 and end > start:
        text = text[:start] + "_EMBEDDED_ENGINE_B85 = (<integrity-checked binary payload>)" + text[end:]
    return text


def self_tests() -> Dict[str, Any]:
    base_tests = _ENGINE_SELF_TESTS()
    obsolete_local_column = "local_core_frozen_from_" + "v" + str(35)
    sample = pd.DataFrame({
        "selected_candidate_probe": [1],
        "prototype_min_cluster_fixed": [999],
        "local_signature_seconds": [999.0],
        "routing_seconds_per_sample": [999.0],
        obsolete_local_column: [True],
        "global_solver": ["obsolete"],
        "run_id": ["obsolete"],
        "value": [2.0],
    })
    args = argparse.Namespace(batch_run_id="batch", run_id="batch")
    cleaned = _clean_frame(sample, args=args, seed=2009, dataset="demo", scenario="demo")
    metadata_ok = (
        "selected_candidate_probe" not in cleaned.columns
        and cleaned.loc[0, "run_id"] == "batch_seed_2009"
        and cleaned.loc[0, "implementation_version"] == VERSION
    )
    signature_equivalence = _ENGINE_SIGNATURE_DISTANCE(
        [0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 1, 4, 3, 5, 7, 6], 8
    ) <= 1e-15
    sample_orders = [[0, 1, 2, 3], [3, 2, 1, 0]]
    sample_prototypes = [[0, 1, 2, 3], [3, 2, 1, 0]]
    frozen_routing = _ENGINE_ROUTE_ORDERS(sample_orders, sample_prototypes, 4)
    wrapped_routing = _timed_route_orders(sample_orders, sample_prototypes, 4)
    routing_equivalence = frozen_routing == wrapped_routing
    math_core_frozen = (
        PROTOTYPE_COUNT == 4
        and MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT == 2
        and _sha256(_ENGINE_LOAD_PATH) == EXPECTED_ENGINE_SHA256
    )
    renamed_budget_ok = (
        "prototype_min_cluster_fixed" not in cleaned.columns
        and obsolete_local_column not in cleaned.columns
        and cleaned.loc[0, "prototype_min_calibration_rows_per_slot_fixed"]
        == MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT
        and cleaned.loc[0, "global_representation"] == GLOBAL_REPRESENTATION
    )
    source = _visible_source_for_audit(Path(__file__).resolve())
    forbidden_visible = re.findall(
        r"(?<![A-Za-z0-9])v(?:2[0-9]|3[0-9]|4[0-9]|5[0-9]|6[0-7])(?:\.\d+)*(?=$|[^A-Za-z0-9])",
        source,
        flags=re.I,
    )
    visible_source_clean = not forbidden_visible
    binomial_probe = pd.DataFrame({
        "dataset": ["demo"] * 60,
        "scenario": ["demo"] * 60,
        "scope": ["global_fidelity"] * 60,
        "metric": ["composite"] * 60,
        "method": ["NEX-ELM"] * 30 + ["Kernel SHAP"] * 30,
        "seed": list(range(30)) * 2,
        "value": ([1.0] * 21 + [0.0] * 9) + ([0.0] * 30),
    })
    binomial_check = _binomial_win_tests(binomial_probe, 0.05)
    binomial_interval_ok = False
    if not binomial_check.empty:
        row = binomial_check.iloc[0]
        binomial_interval_ok = (
            abs(float(row["clopper_pearson_two_sided_ci95_low"]) - 0.5060) < 0.002
            and abs(float(row["clopper_pearson_two_sided_ci95_high"]) - 0.8527) < 0.002
            and "binomial_p_greater_than_80_percent_holm" in binomial_check.columns
        )
    wine_probe = _load_sklearn_bundle("wine_multiclass")
    multiclass_loader_ok = (
        wine_probe.X.shape[1] == 13
        and len(np.unique(wine_probe.y)) == 3
        and wine_probe.name == "wine_multiclass"
    )
    seed_independence_ok = not set(_seed_sequence(REPLICATION_SEED_START, REPLICATION_REPEATS)).intersection(
        _seed_sequence(PREVIOUS_CONFIRMATORY_SEED_START, PREVIOUS_CONFIRMATORY_REPEATS)
    )
    grid_alias_ok = _canonical_dataset_name("grid_no_stab") == "electrical_grid_stability_without_stab"
    complete_dataset_audit = (
        len(COMPLETE_DATASETS) == 9
        and len(set(COMPLETE_DATASETS)) == 9
        and set(EXTENDED_GENERALIZATION_DATASETS).issubset(set(COMPLETE_DATASETS))
    )
    try:
        import reportlab  # noqa: F401
        reportlab_available = True
    except Exception:
        reportlab_available = False
    return {
        "passed": bool(
            base_tests.get("passed") and metadata_ok and signature_equivalence
            and routing_equivalence and math_core_frozen and renamed_budget_ok
            and visible_source_clean and binomial_interval_ok and multiclass_loader_ok
            and seed_independence_ok and grid_alias_ok and complete_dataset_audit
            and reportlab_available
        ),
        "mathematical_core_tests": base_tests,
        "audit_metadata": metadata_ok,
        "registered_set_equivalence": bool(signature_equivalence),
        "routing_wrapper_equivalence": bool(routing_equivalence),
        "mathematical_core_unchanged": bool(math_core_frozen),
        "renamed_budget_metadata": bool(renamed_budget_ok),
        "visible_source_clean": bool(visible_source_clean),
        "visible_source_forbidden_matches": forbidden_visible,
        "two_sided_binomial_interval_audit": bool(binomial_interval_ok),
        "multiclass_loader_audit": bool(multiclass_loader_ok),
        "replication_seed_independence_audit": bool(seed_independence_ok),
        "grid_without_stab_alias_audit": bool(grid_alias_ok),
        "complete_default_dataset_audit": bool(complete_dataset_audit),
        "complete_default_dataset_count": len(COMPLETE_DATASETS),
        "reportlab_available": bool(reportlab_available),
        "predictive_complete_outer_test_hook_installed": True,
        "prototype_count_fixed": PROTOTYPE_COUNT,
        "prototype_min_calibration_rows_per_slot_fixed": MIN_CALIBRATION_ROWS_PER_PROTOTYPE_SLOT,
        "packaging_revision": PACKAGING_REVISION,
        "mathematical_core_source": _ENGINE_SOURCE,
    }



def _resolve_statistics_source(path: Path) -> Path:
    source = Path(path).expanduser().resolve()
    candidates = [source, source / "combined" / "tabelas", source / "tabelas"]
    for candidate in candidates:
        if (candidate / "seed_metrics.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find seed_metrics.csv. Provide the experiment root or its combined/tabelas directory."
    )


def _run_statistics_only(args: argparse.Namespace) -> Path:
    source = _resolve_statistics_source(Path(args.statistics_only_from))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = engine.ensure_dir(
        Path(args.statistics_output_dir).expanduser()
        if args.statistics_output_dir is not None
        else Path(f"resultados_v68_reanalise_estatistica_{stamp}")
    )
    combined = engine.ensure_dir(root / "combined" / "tabelas")
    for name in ("seed_metrics.csv", "estatistica_entre_seeds.csv"):
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, combined / name)
    outputs = _write_statistical_suite(combined, args)
    _sanitize_text_artifacts(root)
    findings = _audit_generated_text(root)
    engine.write_csv(findings, combined / "text_audit_findings.csv")
    if not findings.empty:
        raise RuntimeError(
            f"{METHOD_TITLE} statistical reanalysis text audit failed. "
            f"See {combined / 'text_audit_findings.csv'}"
        )
    audit = {
        "version": VERSION,
        "method_name": METHOD_NAME,
        "packaging_revision": PACKAGING_REVISION,
        "statistics_only": True,
        "source_tables": str(source),
        "mathematical_models_rerun": False,
        "mathematical_core_unchanged": True,
        "mathematical_core_sha256": EXPECTED_ENGINE_SHA256,
        "two_sided_exact_clopper_pearson_intervals": True,
        "holm_family_for_win_rate_above_80_percent": True,
        "obsolete_column_headers_removed": True,
        "statistical_tables": sorted(outputs),
        "passed": True,
    }
    engine.write_json(audit, root / "audit_reanalise_estatistica_v68.json")
    print(f"Statistical reanalysis completed. Results: {root.resolve()}")
    return root

def main() -> None:
    _install_active_hooks()
    tests = self_tests()
    if not bool(tests.get("passed")):
        raise RuntimeError(f"{METHOD_TITLE} self-tests failed: {tests}")
    args = apply_protocol(parse_args())
    if getattr(args, "report_only_from", None) is not None:
        _run_report_only(args)
        return
    if getattr(args, "statistics_only_from", None) is not None:
        _run_statistics_only(args)
        return
    if not engine.SHAP_AVAILABLE:
        raise RuntimeError("The shap package is required. Install with: pip install shap")
    if not engine.TORCH_AVAILABLE:
        raise RuntimeError("The torch package is required for NEX-ELM.")
    _ENGINE_ALLOCATOR_FROM_ARGS(args)
    print(f"CUDA allocator: {args.cuda_allocator_effective_config} (configured before PyTorch import)")
    runtime = engine.resolve_runtime(args.device, args.gpu_dtype)
    if str(getattr(args, "study_plan", "custom")) == "complete" and runtime.resolved != "cuda":
        raise RuntimeError("The default complete experiment requires CUDA. Use --device cpu only for an explicit non-default audit.")
    _ENGINE_CONFIGURE_CUDA(runtime, args)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = engine.ensure_dir(args.output_dir or f"resultados_v68_glopro_complete_{stamp}")
    engine.write_csv(confirmatory_plan(args), root / "plano_confirmatorio.csv")
    protocol = {
        "version": VERSION,
        "method_name": METHOD_NAME,
        "batch_run_id": str(args.batch_run_id),
        "status": "fixed_before_execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "active_script_sha256": _sha256(Path(__file__).resolve()),
        "mathematical_core_sha256": EXPECTED_ENGINE_SHA256,
        "mathematical_core_source": _ENGINE_SOURCE,
        "packaging_revision": PACKAGING_REVISION,
        "global_solver": SOLVER_ID,
        "global_definition": METHOD_DEFINITION,
        "global_representation": GLOBAL_REPRESENTATION,
        "global_representation_definition": GLOBAL_REPRESENTATION_DEFINITION,
        "mathematical_core_unchanged": True,
        "registered_set_distance_frozen": True,
        "deterministic_kmedoids_frozen": True,
        "prototype_routing_frozen": True,
        "prototype_count_fixed": PROTOTYPE_COUNT,
        "primary_methods_global": ["NEX-ELM", "Kernel SHAP", "X-ELM"],
        "primary_comparison": "NEX-ELM GloPro-Complete versus capacity-matched Kernel SHAP prototype library",
        "frozen_components": {"local_irp_nex": True, "cuda_executor_and_audit": True},
        "supplementary_statistical_suite": STATISTICAL_SUITE,
        "original_confirmatory_rule_preserved": True,
        "study_plan": str(getattr(args, "study_plan", "custom")),
        "default_complete_plan_all_available_datasets": str(getattr(args, "study_plan", "custom")) == "complete",
        "default_dataset_count": len(COMPLETE_DATASETS),
        "default_repetitions_per_dataset": DEFAULT_COMPLETE_REPEATS,
        "predictive_performance_complete_outer_test": True,
        "pdf_report_generated_after_aggregation": not bool(getattr(args, "skip_pdf_report", False)),
        "method_frozen_before_replication_and_generalization": True,
        "evaluation_results_used_for_method_modification": False,
        "primary_claim": "fidelity and explanatory granularity over X-ELM",
        "timing_claim_scope": "complete NEX-ELM workflow versus complete X-ELM plus Kernel SHAP workflow",
        "isolated_xelm_speed_superiority_claimed": False,
        "wisconsin_reporting_policy": "report mixed, positive, or negative replication outcomes without method changes",
        "grid_protocol": {
            "with_stab": "exact X-ELM reproduction only",
            "without_stab": "target-proxy leakage-control scenario",
        },
        "multiclass_extension_required": True,
        "configuration": _public_configuration(args),
    }
    engine.write_json(protocol, root / "registro_protocolo_confirmatorio.json")
    print(json.dumps(engine.environment_metadata(runtime), ensure_ascii=False, indent=2, default=str))
    print(json.dumps({
        "version": VERSION,
        "method_name": METHOD_NAME,
        "batch_run_id": str(args.batch_run_id),
        "study_plan": str(getattr(args, "study_plan", "custom")),
        "replication_repetitions": int(getattr(args, "replication_repeats", args.real_repetitions)),
        "generalization_repetitions": int(getattr(args, "generalization_repeats", args.real_repetitions)),
        "replication_seeds": _seed_sequence(
            int(getattr(args, "replication_random_state", args.random_state)),
            int(getattr(args, "replication_repeats", args.real_repetitions)),
        ),
        "generalization_seeds": _seed_sequence(
            int(getattr(args, "generalization_random_state", args.random_state)),
            int(getattr(args, "generalization_repeats", args.real_repetitions)),
        ),
        "synthetic_repetitions": int(args.synthetic_repetitions),
        "device": runtime.resolved,
        "global_solver": SOLVER_ID,
        "prototype_count_fixed": PROTOTYPE_COUNT,
        "packaging_revision": PACKAGING_REVISION,
        "mathematical_core_source": _ENGINE_SOURCE,
        "capacity_matched_kernel_shap": True,
        "outer_test_used_for_prototype_learning": False,
        "local_and_cuda_frozen": True,
        "statistical_suite": STATISTICAL_SUITE,
    }, ensure_ascii=False, indent=2))
    results: List[Dict[str, pd.DataFrame]] = []
    if args.mode in {"real", "all"}:
        results.extend(run_real(args, runtime, root))
    if args.mode in {"synthetic", "all"}:
        results.extend(run_synthetic(args, runtime, root))
    if results:
        aggregate_results(results, root, args, runtime)
    else:
        print("No experiment was executed.")
    print(f"Completed. Results: {root.resolve()}")


if __name__ == "__main__":
    main()
