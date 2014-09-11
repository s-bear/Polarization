(* ::Package:: *)

BeginPackage["Polarization`"]


(* Basic Jones Matrices *)

Jrot::usage="Jrot[\[Theta]] gives the Jones matrix of a rotator.
\[Theta] is the rotation angle in radians."

Jpol::usage = 
"Jpol[px,py,\[Theta]] gives the Jones matrix of a rotated linear diattenuator.
px and py are the x and y attenuation coefficients (before rotation);
\[Theta] (optional, default=0) is the rotation angle in radians."

Jret::usage = "Jret[\[Phi],\[Theta]] gives the Jones matrix of a rotated linear retarder.
\[Phi] is the retardance in radians;
\[Theta] (optional, default=0) is the rotation angle in radians."


(* Basic Mueller Matrices *)

Mrot::usage="Mrot[\[Theta]] gives the Mueller matrix of a rotator.
\[Theta] is the rotation angle in radians."

Mpol::usage="Mpol[px,py,\[Theta]] gives the Mueller matrix of a rotated linear diattenuator.
px and py are the x and y amplitude attenuation coefficients;
\[Theta] (optional, default=0) is the rotation angle in radians."

Mret::usage="Mret[\[Phi],\[Theta]] gives the Mueller matrix of a rotated linear retarder.
\[Phi] is the retardance in radians;
\[Theta] (optional, default=0) is the rotation angle in radians."

MvalidQ::usage="MvalidQ[M,\[Epsilon]] tests a Mueller matrix for:
no light amplification in forward or reverse directions;
no negative gains in forward or reverse directions;
and no over-polarization.
M is the Mueller matrix to test;
\[Epsilon] (optional, default=0) is the precision used for comparisons."


(* Stokes vector properties *)

DoP::usage="DoP[S] gives the total degree of polarization of the given Stokes vector."

DoLP::usage="DoLP[S] gives the degree of linear polarization of the given Stokes vector."

DoCP::usage="DoCP[S] gives the degree of circular polarization of the given Stokes vector."

AoP::usage="AoP[S] gives the angle of polarization in radians of the given Stokes or Jones vector."

ElP::usage="ElP[S] gives the ellipticity angle in radians of the given Stokes or Jones vector."


(* Jones to Mueller/Stokes conversion *)

JtoS::usage="JtoS[E] gives the Stokes vector analagous to the given Jones vector."

JtoM::usage="JtoM[J] gives the Mueller matrix analagous to the given Jones matrix J."


(* Fresnel Reflection & Refraction *)

Reflect::usage="Reflect[ki,n] gives the unit propagation vector of a reflected wave.
ki is the unit propagation vector of the incident wave;
n is the unit normal vector of the reflecting surface."

Refract::usage="Refract[ki,n,mi,mt] gives the unit propagation vector of a refracted/transmitted wave according to Snell's law. 
ki is the unit propagation vector of the incident wave;
n is the unit normal vector of the reflecting interface (pointing from the transmission side to the incident side);
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side."


(* Fresnel Reflection & Refraction Jones Matrices *)

Jfr::usage = "Jfr[\[Theta]i,mi,mt] gives the Jones matrix of the reflection of a transverse plane wave off an interface between two isotropic media. The x component corresponds to s-polarization, y to p-polarization.
\[Theta]i is the angle of incidence in radians (measured from the negative normal vector);
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side."

Jfrv::usage="Jfrv[ki,n,mi,mt] gives the Jones matrix of the reflection of a transverse plane wave off the interface between two isotropic media. The x component corresponds to s-polarization and y to p-polarization. 
ki is the unit propagation vector of the incident wave;
n is the unit normal vector of the interface, pointing from the transmission side to the incident side;
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side material."

Jft::usage="Jft[\[Theta]i,mi,mt] gives the Jones matrix of the transmission/refraction of a transverse plane wave through an interface between two isotropic media. The x component corresponds to s-polarization, y to p-polarization.
\[Theta]i is the angle of incidence in radians (measured from the negative normal vector);
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side material."

Jftv::usage="Jftv[ki,n,mi,mt] gives the Jones matrix of the transmission/refraction of a transverse plane wave through the interface between two isotropic media. The x component corresponds to s-polarization and y to p-polarization.
ki is the unit propagation vector of the incident wave;
n is the unit normal vector of the interface, pointing from the transmission side to the incident side;
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side material."


(* Fresnel Reflection & Refraction Mueller Matrices *)

Mfr::usage = "Mfr[\[Theta]i,mi,mt] gives the Mueller matrix of the reflection of a transverse plane wave off an interface between two isotropic media. The x component corresponds to s-polarization, y to p-polarization.
\[Theta]i is the angle of incidence in radians (measured from the negative normal vector);
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side."

Mfrv::usage="Mfrv[ki,n,mi,mt] gives the Mueller matrix of the reflection of a transverse plane wave off the interface between two isotropic media. The x component corresponds to s-polarization and y to p-polarization. 
ki is the unit propagation vector of the incident wave;
n is the unit normal vector of the interface, pointing from the transmission side to the incident side;
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side material."

Mft::usage="Mft[\[Theta]i,mi,mt] gives the Mueller matrix of the transmission/refraction of a transverse plane wave through an interface between two isotropic media. The x component corresponds to s-polarization, y to p-polarization.
\[Theta]i is the angle of incidence in radians (measured from the negative normal vector);
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side material."

Mftv::usage="Mftv[ki,n,mi,mt] gives the Mueller matrix of the transmission/refraction of a transverse plane wave through the interface between two isotropic media. The x component corresponds to s-polarization and y to p-polarization.
ki is the unit propagation vector of the incident wave;
n is the unit normal vector of the interface, pointing from the transmission side to the incident side;
mi is the (complex) index of refraction of the incident side material;
mt is the (complex) index of refraction of the transmission side material."


Begin["`Private`"]


(* Basic Jones Matrices *)

Jrot[\[Theta]_]:=({
 {Cos[\[Theta]], Sin[\[Theta]]},
 {-Sin[\[Theta]], Cos[\[Theta]]}
})

Jpol[px_,py_,\[Theta]_:0]:=Jrot[-\[Theta]].({
 {px, 0},
 {0, py}
}).Jrot[\[Theta]]

Jret[\[Phi]_,\[Theta]_:0]:=Jrot[-\[Theta]].({
 {Exp[I*\[Phi]/2], 0},
 {0, Exp[-I*\[Phi]/2]}
}).Jrot[\[Theta]]


(* Basic Mueller Matrices *)

Mrot[\[Theta]_]:=\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{"1", "0", "0", "0"},
{"0", 
RowBox[{"Cos", "[", 
RowBox[{"2", " ", "\[Theta]"}], "]"}], 
RowBox[{"Sin", "[", 
RowBox[{"2", " ", "\[Theta]"}], "]"}], "0"},
{"0", 
RowBox[{"-", 
RowBox[{"Sin", "[", 
RowBox[{"2", " ", "\[Theta]"}], "]"}]}], 
RowBox[{"Cos", "[", 
RowBox[{"2", " ", "\[Theta]"}], "]"}], "0"},
{"0", "0", "0", "1"}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\)

Mpol[px_,py_,\[Theta]_:0]:=
1/2 Mrot[-\[Theta]].\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{
RowBox[{
SuperscriptBox["px", "2"], "+", 
SuperscriptBox["py", "2"]}], 
RowBox[{
SuperscriptBox["px", "2"], "-", 
SuperscriptBox["py", "2"]}], "0", "0"},
{
RowBox[{
SuperscriptBox["px", "2"], "-", 
SuperscriptBox["py", "2"]}], 
RowBox[{
SuperscriptBox["px", "2"], "+", 
SuperscriptBox["py", "2"]}], "0", "0"},
{"0", "0", 
RowBox[{"2", " ", "px", " ", "py"}], "0"},
{"0", "0", "0", 
RowBox[{"2", "px", " ", "py"}]}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\).Mrot[\[Theta]]

Mret[\[Phi]_,\[Theta]_:0]:=
Mrot[-\[Theta]].\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{"1", "0", "0", "0"},
{"0", "1", "0", "0"},
{"0", "0", 
RowBox[{"Cos", "[", "\[Phi]", "]"}], 
RowBox[{"Sin", "[", "\[Phi]", "]"}]},
{"0", "0", 
RowBox[{"-", 
RowBox[{"Sin", "[", "\[Phi]", "]"}]}], 
RowBox[{"Cos", "[", "\[Phi]", "]"}]}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\).Mrot[\[Theta]]

MvalidQ[m_,eps_:0] := Module[{gf,gr,He},
gf = m[[1,1]] + Norm[m[[1,2;;4]]];
gr = m[[1,1]] + Norm[m[[2;;4,1]]];
He = Eigenvalues[
1/4 Sum[
m[[i+1,j+1]]* KroneckerProduct[PauliMatrix[i],PauliMatrix[j]],
{i,0,3},{j,0,3}
]];
0 <= gf <=1 && 0<= gr <=1  && And @@ Thread[He + eps >= 0]
]



(* Stokes vector properties *)
DoP[S_]:=Norm[S[[2;;4]]]/S[[1]]
DoLP[S_]:=Norm[S[[2;;3]]]/S[[1]]
DoCP[S_]:=Abs[S[[4]]]/S[[1]]
AoP[ES_]:=Switch[Length[ES],
2,
4,1/2 ArcTan[S[[2]],S[[3]]],
_,Message[AoP::vlen,"Input vector must be length 2 or length 4."]
]
ElP[ES_]:=1/2 ArcSin[S[[4]]/S[[1]]]


(* Jones to Mueller/Stokes conversion *)

JtoS[E_]:=({
 {1, 0, 0, 1},
 {1, 0, 0, -1},
 {0, 1, 1, 0},
 {0, I, -I, 0}
}).KroneckerProduct[E,E\[Conjugate]]

JtoM[J_]:=1/2 ({
 {1, 0, 0, 1},
 {1, 0, 0, -1},
 {0, 1, 1, 0},
 {0, I, -I, 0}
}).KroneckerProduct[J,J\[Conjugate]].({
 {1, 1, 0, 0},
 {0, 0, 1, -I},
 {0, 0, 1, I},
 {1, -1, 0, 0}
})


(* Fresnel Reflection & Refraction *)

Reflect[ki_,n_]:=ki-2*Dot[n,ki]*n

Refract[ki_,n_,mi_,mt_]:=
Module[{st},
st=(mi/mt)((n\[Cross]ki)\[Cross]n); (*st is in the s direction w/ magnitude sin(\[Theta]i) *)
st-n Sqrt[1-Norm[st]^2]
]


(* Fresnel Reflection & Refraction Jones Matrices *)

Jfr[\[Theta]i_,mi_,mt_]:=Module[{c\[Theta]i,c\[Theta]t},
c\[Theta]i=Cos[\[Theta]i];
c\[Theta]t=Sqrt[1-(mi/mt)^2Sin[\[Theta]i]^2];
({
 {(mi c\[Theta]i - mt c\[Theta]t)/(mi c\[Theta]i + mt c\[Theta]t), 0},
 {0, (mt c\[Theta]i - mi c\[Theta]t)/(mt c\[Theta]i + mi c\[Theta]t)}
})
]

Jfrv[ki_,n_,mi_,mt_]:=Module[{kt},
kt=Refract[ki,n,mi,mt];
({
 {(mi ki - mt kt).n/(mi ki + mt kt).n, 0},
 {0, (mt ki - mi kt).n/(mt ki + mi kt).n}
})
]

Jft[\[Theta]i_,mi_,mt_]:=Module[{c\[Theta]i,c\[Theta]t},
c\[Theta]i=Cos[\[Theta]i];
c\[Theta]t=Sqrt[1-(mi/mt)^2Sin[\[Theta]i]^2];
({
 {(2 mi c\[Theta]i)/(mi c\[Theta]i + mt c\[Theta]t), 0},
 {0, (2 mi c\[Theta]i)/(mt c\[Theta]i + mi c\[Theta]t)}
})
]
Jftv[ki_,n_,mi_,mt_]:=Module[{kt},
kt=Refract[ki,n,mi,mt];
({
 {(2 mi ki.n)/(mi ki+mt kt).n, 0},
 {0, (2 mi ki.n)/(mt ki +mi kt).n}
})
]


(* Fresnel Reflection & Refraction Mueller Matrices *)

Mfr[\[Theta]i_,mi_,mt_]:=Module[{c\[Theta]i,c\[Theta]t,rs,rp},
c\[Theta]i=Cos[\[Theta]i];
c\[Theta]t=Sqrt[1-(mi/mt)^2Sin[\[Theta]i]^2];
rs=(mi c\[Theta]i - mt c\[Theta]t)/(mi c\[Theta]i + mt c\[Theta]t);
rp=(mt c\[Theta]i - mi c\[Theta]t)/(mt c\[Theta]i + mi c\[Theta]t);
Mpol[Abs[rs],Abs[rp]].Mret[Arg[rs rp\[Conjugate]]]
]

Mfrv[ki_,n_,mi_,mt_]:=Module[{kt,rs,rp},
kt=Refract[ki,n,mi,mt];
rs=(mi ki - mt kt).n/(mi ki + mt kt).n;
rp=(mt ki - mi kt).n/(mt ki + mi kt).n;
Mpol[Abs[rs],Abs[rp]].Mret[Arg[rs rp\[Conjugate]]]
]

Mft[\[Theta]i_,mi_,mt_]:=Module[{c\[Theta]i,c\[Theta]t,ts,tp},
c\[Theta]i=Cos[\[Theta]i];
c\[Theta]t=Sqrt[1-(mi/mt)^2Sin[\[Theta]i]^2];
ts=(2 mi c\[Theta]i)/(mi c\[Theta]i + mt c\[Theta]t);
tp=(2 mi c\[Theta]i)/(mt c\[Theta]i + mi c\[Theta]t);
Mpol[Abs[ts],Abs[tp]].Mret[Arg[ts tp\[Conjugate]]]
]

Mftv[ki_,n_,mi_,mt_]:=Module[{kt,ts,tp},
kt=Refract[ki,n,mi,mt];
ts=(2 mi ki.n)/(mi ki+mt kt).n;
tp=(2 mi ki.n)/(mt ki +mi kt).n;
Mpol[Abs[ts],Abs[tp]].Mret[Arg[ts tp\[Conjugate]]]
]


End[]
EndPackage[]
