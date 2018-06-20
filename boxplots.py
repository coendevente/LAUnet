import numpy as np
import matplotlib.pyplot as plt

x_without_post = [
    [0.8591827723635188, 0.85439837915575, 0.8516896814684601, 0.8075452129383548, 0.8403924371519511, 0.7574217456712534, 0.848855504503233, 0.8401735289477835, 0.7269754136384723, 0.8631326309299688, 0.8370841841679181, 0.10013027634949305, 0.8420613446245998, 0.7821345732687474, 0.8248279910630664, 0.7519473270240784, 0.7735543199060747, 0.8110735716807019, 0.8791757636996806, 0.8743669755157897, 0.8364451873730707, 0.7493481542472897, 0.8095868156162594, 0.8389086739362934, 0.8363035509664023]
    , [0.8701087499659305, 0.9026791953750816, 0.8819196619750478, 0.8189510931014699, 0.8971456344581236, 0.8104808731561948, 0.878280463020235, 0.8610808238213558, 0.8749487855428333, 0.8626163944435964, 0.9217502973471463, 0.09893460313110206, 0.8754964857879053, 0.8801681907754525, 0.9006765221748935, 0.8503433140985157, 0.8519857594498674, 0.8874175773567322, 0.8998189217777938, 0.9086003463226707, 0.8499738285869376, 0.9127782014837412, 0.9016166847464744, 0.8054612354033813, 0.9269714595146793]
    , [0.8764743298500681, 0.8481234726162599, 0.8772708152588269, 0.8430930397727273, 0.8542425026024573, 0.7796194835517951, 0.8679047456783004, 0.8624053594652278, 0.881999286083849, 0.8474378710727529, 0.8650238300717623, 0.038683703833362504, 0.9380725818217647, 0.8238324442854699, 0.7934777932833522, 0.8691766881716713, 0.8678016076744783, 0.8647063427870691, 0.8345585182183276, 0.8854355499435325, 0.9255841307298813, 0.9021323460493017, 0.9036205588835596, 0.8507887958340193, 0.9224921788320019]
    , [0.8769427769228059, 0.8866042712046617, 0.8513872496537813, 0.7835580515078796, 0.8683814618755136, 0.6536707342744328, 0.8752929770878835, 0.8454953749528468, 0.8546446323808877, 0.8776607327562742, 0.8810766988292916, 0.031012562324571214, 0.8961374267214086, 0.8698548955193196, 0.8813666040457373, 0.8209307823116283, 0.814967176735751, 0.8528305317831043, 0.8895548134551164, 0.9058411679138817, 0.8389537785764201, 0.884799704213535, 0.8752293075596199, 0.8157053882534129, 0.9116020399147421]
    , [0.8702576112412178, 0.8679648526976771, 0.8766977039609714, 0.8493886780882297, 0.83296640591399, 0.7313608025955906, 0.8548288815013074, 0.8578385454373528, 0.8862036761053155, 0.9005076924394065, 0.9136019700831781, 0.14191847522714893, 0.8975043885980044, 0.8389745491025669, 0.8880677484192221, 0.8685084252353329, 0.856999552171966, 0.8674513211422252, 0.8384029088874904, 0.8777153304719623, 0.8631151207873546, 0.9109760653079074, 0.9014232335530459, 0.866221980919879, 0.9242170125537182]
    , [0.8876432605352891, 0.8924817013835024, 0.9037009011487928, 0.7796398696967196, 0.8911534620315796, 0.7773897442551919, 0.8798204033103362, 0.8589291439563963, 0.8965764763051188, 0.8724274238880124, 0.9183297977358174, 0.0, 0.898491818142736, 0.8812629071798272, 0.9101003249678575, 0.8743292233684442, 0.8682899760158578, 0.865805473575564, 0.8648888858267852, 0.9208388796240963, 0.8854831285760795, 0.9028833005822746, 0.9132742685307665, 0.8321739929412655, 0.9224663129910055]
]

x_with_post = [
    [0.8638299972994241, 0.854942280269934, 0.8621593310401572, 0.8079900597811258, 0.8425707015105849,
     0.7662159859448724, 0.8628876277923017, 0.8439580989207778, 0.8121000984539485, 0.8673605171666907,
     0.840482162655197, 0.0, 0.8539718619947858, 0.7874474842398411, 0.8311193525979168, 0.7633050648048763,
     0.7764894857187381, 0.823306844398872, 0.8868134880745682, 0.8858775827495737, 0.8673328522573965,
     0.8629776379307361, 0.8285891998315229, 0.8421734293447277, 0.8368806018889878],
[0.8705437489774028, 0.9034879822421586, 0.8824991429550908, 0.8194702680368785, 0.8976814789808045, 0.8119824494419999, 0.8787245034686058, 0.8622996549003356, 0.9021551422130083, 0.864600235380962, 0.9243794081039967, 0.07760748925634949, 0.8768160019199656, 0.8809572880685238, 0.9029937763114649, 0.8533109928493088, 0.8524542936288089, 0.8896493286403337, 0.9004711446338671, 0.9089582503368352, 0.8527229479916059, 0.9279360559369582, 0.9023923088878139, 0.805892675882111, 0.9179697204968944]
,
    [0.8779641657198753, 0.8515243425623723, 0.8948364095792045, 0.8496079412933335, 0.8584024665175588,
     0.7864830513415502, 0.8702428511354079, 0.8653675636663574, 0.8833349400687682, 0.8503581097768396,
     0.8667135370311584, 0.0, 0.9403621803760082, 0.8331976737327624, 0.795654657043324, 0.870155564143532,
     0.8689190383825357, 0.8728686335550955, 0.8403170780654224, 0.8865716544928407, 0.9253495648047924,
     0.904041969603036, 0.9055609479361956, 0.8519585623702279, 0.9209451205887172]
,
[0.8843171359742564, 0.8868125811212062, 0.853981665297764, 0.7820986216364907, 0.8717078854099356, 0.6307248349376068, 0.8766063788999252, 0.8456672573620586, 0.8642772173479861, 0.8808075013487949, 0.9114543294430079, 0.015389314753669144, 0.8962674081050217, 0.8718209648119492, 0.8993863453950626, 0.8222136003006132, 0.814877910278251, 0.8528305317831043, 0.8875289973842232, 0.9058411679138817, 0.8413121738886686, 0.9046212449809902, 0.8859162703902409, 0.8164617706207715, 0.9132048901622927]
,
    [0.8702576112412178, 0.8723999319409588, 0.8694188174589921, 0.8496345383758667, 0.832842151885796,
     0.7304168193782694, 0.8548288815013074, 0.8594904775440592, 0.8865806820982991, 0.9006214238161802,
     0.9142043775369268, 0.12279345885796748, 0.9009860202544199, 0.8482457526524102, 0.9094219493884999,
     0.8808927921981707, 0.8578666939904708, 0.8745345572569244, 0.8413978853942496, 0.877735120432413,
     0.8632245363638398, 0.9110205494898506, 0.9048021390599683, 0.8662667735346589, 0.9253067548575044]
,
    [0.888069192157922, 0.8917725202862328, 0.894560820421729, 0.7745376659935133, 0.890505393230092,
     0.7791524829512684, 0.8798204033103362, 0.8608641387765651, 0.8972730171157828, 0.8718161645588576,
     0.9183297977358174, 0.0, 0.8975198876930276, 0.8820687208231509, 0.9111346757041161, 0.8757167611186915,
     0.8605282632974072, 0.8652382234185734, 0.8665522990310968, 0.9208388796240963, 0.8861712346851073,
     0.9050859997140529, 0.9132956336251906, 0.8292132202716636, 0.9214766213899999]
]

plt.figure()
plt.subplot(2, 2, 1)
plt.boxplot(x_with_post, widths=.8)
plt.subplot(2, 2, 2)
plt.boxplot(x_with_post, widths=.8)
plt.subplot(2, 2, 3)
plt.boxplot(x_with_post, widths=.7, sym='+')
plt.subplot(2, 2, 4)
plt.boxplot(x_with_post, widths=.8)
plt.show()