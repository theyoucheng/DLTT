#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char* argv[]) {
  float finput[784];
  finput[0]=0.0;
  finput[1]=0.0;
  finput[2]=0.0;
  finput[3]=0.0;
  finput[4]=0.0;
  finput[5]=0.0;
  finput[6]=0.0;
  finput[7]=0.0;
  finput[8]=0.0;
  finput[9]=0.0;
  finput[10]=0.0;
  finput[11]=0.0;
  finput[12]=0.0;
  finput[13]=0.0;
  finput[14]=0.0;
  finput[15]=0.0;
  finput[16]=0.0;
  finput[17]=0.0;
  finput[18]=0.0;
  finput[19]=0.0;
  finput[20]=0.0;
  finput[21]=0.0;
  finput[22]=0.0;
  finput[23]=0.0;
  finput[24]=0.0;
  finput[25]=0.0;
  finput[26]=0.0;
  finput[27]=0.0;
  finput[28]=0.0;
  finput[29]=0.0;
  finput[30]=0.0;
  finput[31]=0.0;
  finput[32]=0.0;
  finput[33]=0.0;
  finput[34]=0.0;
  finput[35]=0.0;
  finput[36]=0.0;
  finput[37]=0.0;
  finput[38]=0.0;
  finput[39]=0.0;
  finput[40]=0.0;
  finput[41]=0.0;
  finput[42]=0.0;
  finput[43]=0.0;
  finput[44]=0.0;
  finput[45]=0.0;
  finput[46]=0.0;
  finput[47]=0.0;
  finput[48]=0.0;
  finput[49]=0.0;
  finput[50]=0.0;
  finput[51]=0.0;
  finput[52]=0.0;
  finput[53]=0.0;
  finput[54]=0.0;
  finput[55]=0.0;
  finput[56]=0.0;
  finput[57]=0.0;
  finput[58]=0.0;
  finput[59]=0.0;
  finput[60]=0.0;
  finput[61]=0.0;
  finput[62]=0.0;
  finput[63]=0.0;
  finput[64]=0.0;
  finput[65]=0.0;
  finput[66]=0.0;
  finput[67]=0.0;
  finput[68]=0.0;
  finput[69]=0.0;
  finput[70]=0.0;
  finput[71]=0.0;
  finput[72]=0.0;
  finput[73]=0.0;
  finput[74]=0.0;
  finput[75]=0.0;
  finput[76]=0.0;
  finput[77]=0.0;
  finput[78]=0.0;
  finput[79]=0.0;
  finput[80]=0.0;
  finput[81]=0.0;
  finput[82]=0.0;
  finput[83]=0.0;
  finput[84]=0.0;
  finput[85]=0.0;
  finput[86]=0.0;
  finput[87]=0.0;
  finput[88]=0.0;
  finput[89]=0.0;
  finput[90]=0.0;
  finput[91]=0.0;
  finput[92]=0.0;
  finput[93]=0.0;
  finput[94]=0.0;
  finput[95]=0.0;
  finput[96]=0.0;
  finput[97]=0.0;
  finput[98]=0.0;
  finput[99]=0.0;
  finput[100]=0.0;
  finput[101]=0.0;
  finput[102]=0.0;
  finput[103]=0.0;
  finput[104]=0.0;
  finput[105]=0.0;
  finput[106]=0.0;
  finput[107]=0.0;
  finput[108]=0.0;
  finput[109]=0.0;
  finput[110]=0.0;
  finput[111]=0.0;
  finput[112]=0.0;
  finput[113]=0.0;
  finput[114]=0.0;
  finput[115]=0.0;
  finput[116]=0.0;
  finput[117]=0.0;
  finput[118]=0.0;
  finput[119]=0.0;
  finput[120]=0.0;
  finput[121]=0.0;
  finput[122]=0.0;
  finput[123]=0.16470588235294117;
  finput[124]=0.4627450980392157;
  finput[125]=0.8588235294117647;
  finput[126]=0.6509803921568628;
  finput[127]=0.4627450980392157;
  finput[128]=0.4627450980392157;
  finput[129]=0.023529411764705882;
  finput[130]=0.0;
  finput[131]=0.0;
  finput[132]=0.0;
  finput[133]=0.0;
  finput[134]=0.0;
  finput[135]=0.0;
  finput[136]=0.0;
  finput[137]=0.0;
  finput[138]=0.0;
  finput[139]=0.0;
  finput[140]=0.0;
  finput[141]=0.0;
  finput[142]=0.0;
  finput[143]=0.0;
  finput[144]=0.0;
  finput[145]=0.0;
  finput[146]=0.0;
  finput[147]=0.0;
  finput[148]=0.0;
  finput[149]=0.0;
  finput[150]=0.403921568627451;
  finput[151]=0.9490196078431372;
  finput[152]=0.996078431372549;
  finput[153]=0.996078431372549;
  finput[154]=0.996078431372549;
  finput[155]=0.996078431372549;
  finput[156]=0.996078431372549;
  finput[157]=0.25882352941176473;
  finput[158]=0.0;
  finput[159]=0.0;
  finput[160]=0.0;
  finput[161]=0.0;
  finput[162]=0.0;
  finput[163]=0.0;
  finput[164]=0.0;
  finput[165]=0.0;
  finput[166]=0.0;
  finput[167]=0.0;
  finput[168]=0.0;
  finput[169]=0.0;
  finput[170]=0.0;
  finput[171]=0.0;
  finput[172]=0.0;
  finput[173]=0.0;
  finput[174]=0.0;
  finput[175]=0.0;
  finput[176]=0.0;
  finput[177]=0.0;
  finput[178]=0.07058823529411765;
  finput[179]=0.9098039215686274;
  finput[180]=0.996078431372549;
  finput[181]=0.996078431372549;
  finput[182]=0.996078431372549;
  finput[183]=0.996078431372549;
  finput[184]=0.996078431372549;
  finput[185]=0.9333333333333333;
  finput[186]=0.27450980392156865;
  finput[187]=0.0;
  finput[188]=0.0;
  finput[189]=0.0;
  finput[190]=0.0;
  finput[191]=0.0;
  finput[192]=0.0;
  finput[193]=0.0;
  finput[194]=0.0;
  finput[195]=0.0;
  finput[196]=0.0;
  finput[197]=0.0;
  finput[198]=0.0;
  finput[199]=0.0;
  finput[200]=0.0;
  finput[201]=0.0;
  finput[202]=0.0;
  finput[203]=0.0;
  finput[204]=0.0;
  finput[205]=0.0;
  finput[206]=0.0;
  finput[207]=0.40784313725490196;
  finput[208]=0.9568627450980393;
  finput[209]=0.996078431372549;
  finput[210]=0.8784313725490196;
  finput[211]=0.996078431372549;
  finput[212]=0.996078431372549;
  finput[213]=0.996078431372549;
  finput[214]=0.5529411764705883;
  finput[215]=0.0;
  finput[216]=0.0;
  finput[217]=0.0;
  finput[218]=0.0;
  finput[219]=0.0;
  finput[220]=0.0;
  finput[221]=0.0;
  finput[222]=0.0;
  finput[223]=0.0;
  finput[224]=0.0;
  finput[225]=0.0;
  finput[226]=0.0;
  finput[227]=0.0;
  finput[228]=0.0;
  finput[229]=0.0;
  finput[230]=0.0;
  finput[231]=0.0;
  finput[232]=0.0;
  finput[233]=0.0;
  finput[234]=0.0;
  finput[235]=0.0;
  finput[236]=0.8117647058823529;
  finput[237]=0.996078431372549;
  finput[238]=0.8235294117647058;
  finput[239]=0.996078431372549;
  finput[240]=0.996078431372549;
  finput[241]=0.996078431372549;
  finput[242]=0.13333333333333333;
  finput[243]=0.0;
  finput[244]=0.0;
  finput[245]=0.0;
  finput[246]=0.0;
  finput[247]=0.0;
  finput[248]=0.0;
  finput[249]=0.0;
  finput[250]=0.0;
  finput[251]=0.0;
  finput[252]=0.0;
  finput[253]=0.0;
  finput[254]=0.0;
  finput[255]=0.0;
  finput[256]=0.0;
  finput[257]=0.0;
  finput[258]=0.0;
  finput[259]=0.0;
  finput[260]=0.0;
  finput[261]=0.0;
  finput[262]=0.0;
  finput[263]=0.0;
  finput[264]=0.32941176470588235;
  finput[265]=0.807843137254902;
  finput[266]=0.996078431372549;
  finput[267]=0.996078431372549;
  finput[268]=0.996078431372549;
  finput[269]=0.996078431372549;
  finput[270]=0.1607843137254902;
  finput[271]=0.0;
  finput[272]=0.0;
  finput[273]=0.0;
  finput[274]=0.0;
  finput[275]=0.0;
  finput[276]=0.0;
  finput[277]=0.0;
  finput[278]=0.0;
  finput[279]=0.0;
  finput[280]=0.0;
  finput[281]=0.0;
  finput[282]=0.0;
  finput[283]=0.0;
  finput[284]=0.0;
  finput[285]=0.0;
  finput[286]=0.0;
  finput[287]=0.0;
  finput[288]=0.0;
  finput[289]=0.0;
  finput[290]=0.0;
  finput[291]=0.0;
  finput[292]=0.0;
  finput[293]=0.09411764705882353;
  finput[294]=0.8196078431372549;
  finput[295]=0.996078431372549;
  finput[296]=0.996078431372549;
  finput[297]=0.996078431372549;
  finput[298]=0.6705882352941176;
  finput[299]=0.0;
  finput[300]=0.0;
  finput[301]=0.0;
  finput[302]=0.0;
  finput[303]=0.0;
  finput[304]=0.0;
  finput[305]=0.0;
  finput[306]=0.0;
  finput[307]=0.0;
  finput[308]=0.0;
  finput[309]=0.0;
  finput[310]=0.0;
  finput[311]=0.0;
  finput[312]=0.0;
  finput[313]=0.0;
  finput[314]=0.0;
  finput[315]=0.0;
  finput[316]=0.0;
  finput[317]=0.0;
  finput[318]=0.0;
  finput[319]=0.0;
  finput[320]=0.3568627450980392;
  finput[321]=0.5372549019607843;
  finput[322]=0.9921568627450981;
  finput[323]=0.996078431372549;
  finput[324]=0.996078431372549;
  finput[325]=0.996078431372549;
  finput[326]=0.4392156862745098;
  finput[327]=0.0;
  finput[328]=0.0;
  finput[329]=0.0;
  finput[330]=0.0;
  finput[331]=0.0;
  finput[332]=0.0;
  finput[333]=0.0;
  finput[334]=0.0;
  finput[335]=0.0;
  finput[336]=0.0;
  finput[337]=0.0;
  finput[338]=0.0;
  finput[339]=0.0;
  finput[340]=0.0;
  finput[341]=0.0;
  finput[342]=0.0;
  finput[343]=0.0;
  finput[344]=0.0;
  finput[345]=0.0;
  finput[346]=0.1568627450980392;
  finput[347]=0.8392156862745098;
  finput[348]=0.9803921568627451;
  finput[349]=0.996078431372549;
  finput[350]=0.996078431372549;
  finput[351]=0.996078431372549;
  finput[352]=0.996078431372549;
  finput[353]=0.996078431372549;
  finput[354]=0.13333333333333333;
  finput[355]=0.0;
  finput[356]=0.0;
  finput[357]=0.0;
  finput[358]=0.0;
  finput[359]=0.0;
  finput[360]=0.0;
  finput[361]=0.0;
  finput[362]=0.0;
  finput[363]=0.0;
  finput[364]=0.0;
  finput[365]=0.0;
  finput[366]=0.0;
  finput[367]=0.0;
  finput[368]=0.0;
  finput[369]=0.0;
  finput[370]=0.0;
  finput[371]=0.0;
  finput[372]=0.0;
  finput[373]=0.0;
  finput[374]=0.3176470588235294;
  finput[375]=0.9686274509803922;
  finput[376]=0.996078431372549;
  finput[377]=0.996078431372549;
  finput[378]=0.996078431372549;
  finput[379]=0.996078431372549;
  finput[380]=0.996078431372549;
  finput[381]=0.996078431372549;
  finput[382]=0.5725490196078431;
  finput[383]=0.0;
  finput[384]=0.0;
  finput[385]=0.0;
  finput[386]=0.0;
  finput[387]=0.0;
  finput[388]=0.0;
  finput[389]=0.0;
  finput[390]=0.0;
  finput[391]=0.0;
  finput[392]=0.0;
  finput[393]=0.0;
  finput[394]=0.0;
  finput[395]=0.0;
  finput[396]=0.0;
  finput[397]=0.0;
  finput[398]=0.0;
  finput[399]=0.0;
  finput[400]=0.0;
  finput[401]=0.0;
  finput[402]=0.0;
  finput[403]=0.43137254901960786;
  finput[404]=0.9647058823529412;
  finput[405]=0.996078431372549;
  finput[406]=0.996078431372549;
  finput[407]=0.996078431372549;
  finput[408]=0.996078431372549;
  finput[409]=0.996078431372549;
  finput[410]=0.6705882352941176;
  finput[411]=0.0;
  finput[412]=0.0;
  finput[413]=0.0;
  finput[414]=0.0;
  finput[415]=0.0;
  finput[416]=0.0;
  finput[417]=0.0;
  finput[418]=0.0;
  finput[419]=0.0;
  finput[420]=0.0;
  finput[421]=0.0;
  finput[422]=0.0;
  finput[423]=0.0;
  finput[424]=0.0;
  finput[425]=0.0;
  finput[426]=0.0;
  finput[427]=0.0;
  finput[428]=0.0;
  finput[429]=0.0;
  finput[430]=0.0;
  finput[431]=0.0;
  finput[432]=0.28627450980392155;
  finput[433]=0.34901960784313724;
  finput[434]=0.34901960784313724;
  finput[435]=0.36470588235294116;
  finput[436]=0.9411764705882353;
  finput[437]=0.996078431372549;
  finput[438]=0.6705882352941176;
  finput[439]=0.0;
  finput[440]=0.0;
  finput[441]=0.0;
  finput[442]=0.0;
  finput[443]=0.0;
  finput[444]=0.0;
  finput[445]=0.0;
  finput[446]=0.0;
  finput[447]=0.0;
  finput[448]=0.0;
  finput[449]=0.0;
  finput[450]=0.0;
  finput[451]=0.0;
  finput[452]=0.0;
  finput[453]=0.0;
  finput[454]=0.0;
  finput[455]=0.0;
  finput[456]=0.0;
  finput[457]=0.0;
  finput[458]=0.0;
  finput[459]=0.0;
  finput[460]=0.0;
  finput[461]=0.0;
  finput[462]=0.0;
  finput[463]=0.00392156862745098;
  finput[464]=0.5019607843137255;
  finput[465]=0.996078431372549;
  finput[466]=0.8588235294117647;
  finput[467]=0.12156862745098039;
  finput[468]=0.0;
  finput[469]=0.0;
  finput[470]=0.0;
  finput[471]=0.0;
  finput[472]=0.0;
  finput[473]=0.0;
  finput[474]=0.0;
  finput[475]=0.0;
  finput[476]=0.0;
  finput[477]=0.0;
  finput[478]=0.0;
  finput[479]=0.0;
  finput[480]=0.0;
  finput[481]=0.0;
  finput[482]=0.0;
  finput[483]=0.0;
  finput[484]=0.0;
  finput[485]=0.0;
  finput[486]=0.0;
  finput[487]=0.0;
  finput[488]=0.0;
  finput[489]=0.0;
  finput[490]=0.0;
  finput[491]=0.027450980392156862;
  finput[492]=0.996078431372549;
  finput[493]=0.996078431372549;
  finput[494]=0.8392156862745098;
  finput[495]=0.10980392156862745;
  finput[496]=0.0;
  finput[497]=0.0;
  finput[498]=0.0;
  finput[499]=0.0;
  finput[500]=0.0;
  finput[501]=0.0;
  finput[502]=0.0;
  finput[503]=0.0;
  finput[504]=0.0;
  finput[505]=0.0;
  finput[506]=0.0;
  finput[507]=0.0;
  finput[508]=0.0;
  finput[509]=0.0;
  finput[510]=0.0;
  finput[511]=0.0;
  finput[512]=0.0;
  finput[513]=0.0;
  finput[514]=0.0;
  finput[515]=0.0;
  finput[516]=0.0;
  finput[517]=0.0;
  finput[518]=0.0;
  finput[519]=0.5411764705882353;
  finput[520]=0.996078431372549;
  finput[521]=0.996078431372549;
  finput[522]=0.4549019607843137;
  finput[523]=0.0;
  finput[524]=0.0;
  finput[525]=0.0;
  finput[526]=0.0;
  finput[527]=0.0;
  finput[528]=0.0;
  finput[529]=0.0;
  finput[530]=0.0;
  finput[531]=0.0;
  finput[532]=0.0;
  finput[533]=0.0;
  finput[534]=0.0;
  finput[535]=0.0;
  finput[536]=0.0;
  finput[537]=0.0;
  finput[538]=0.07450980392156863;
  finput[539]=0.6941176470588235;
  finput[540]=0.35294117647058826;
  finput[541]=0.0;
  finput[542]=0.0;
  finput[543]=0.0;
  finput[544]=0.0;
  finput[545]=0.0;
  finput[546]=0.09803921568627451;
  finput[547]=0.9411764705882353;
  finput[548]=0.996078431372549;
  finput[549]=0.996078431372549;
  finput[550]=0.13333333333333333;
  finput[551]=0.0;
  finput[552]=0.0;
  finput[553]=0.0;
  finput[554]=0.0;
  finput[555]=0.0;
  finput[556]=0.0;
  finput[557]=0.0;
  finput[558]=0.0;
  finput[559]=0.0;
  finput[560]=0.0;
  finput[561]=0.0;
  finput[562]=0.0;
  finput[563]=0.0;
  finput[564]=0.0;
  finput[565]=0.0;
  finput[566]=0.6431372549019608;
  finput[567]=0.996078431372549;
  finput[568]=0.8431372549019608;
  finput[569]=0.24705882352941178;
  finput[570]=0.1411764705882353;
  finput[571]=0.0;
  finput[572]=0.2;
  finput[573]=0.34901960784313724;
  finput[574]=0.807843137254902;
  finput[575]=0.996078431372549;
  finput[576]=0.996078431372549;
  finput[577]=0.5450980392156862;
  finput[578]=0.03137254901960784;
  finput[579]=0.0;
  finput[580]=0.0;
  finput[581]=0.0;
  finput[582]=0.0;
  finput[583]=0.0;
  finput[584]=0.0;
  finput[585]=0.0;
  finput[586]=0.0;
  finput[587]=0.0;
  finput[588]=0.0;
  finput[589]=0.0;
  finput[590]=0.0;
  finput[591]=0.0;
  finput[592]=0.0;
  finput[593]=0.0;
  finput[594]=0.2235294117647059;
  finput[595]=0.7725490196078432;
  finput[596]=0.996078431372549;
  finput[597]=0.996078431372549;
  finput[598]=0.8705882352941177;
  finput[599]=0.7058823529411765;
  finput[600]=0.9450980392156862;
  finput[601]=0.996078431372549;
  finput[602]=0.996078431372549;
  finput[603]=0.9921568627450981;
  finput[604]=0.8352941176470589;
  finput[605]=0.043137254901960784;
  finput[606]=0.0;
  finput[607]=0.0;
  finput[608]=0.0;
  finput[609]=0.0;
  finput[610]=0.0;
  finput[611]=0.0;
  finput[612]=0.0;
  finput[613]=0.0;
  finput[614]=0.0;
  finput[615]=0.0;
  finput[616]=0.0;
  finput[617]=0.0;
  finput[618]=0.0;
  finput[619]=0.0;
  finput[620]=0.0;
  finput[621]=0.0;
  finput[622]=0.0;
  finput[623]=0.5490196078431373;
  finput[624]=0.4117647058823529;
  finput[625]=0.996078431372549;
  finput[626]=0.996078431372549;
  finput[627]=0.996078431372549;
  finput[628]=0.996078431372549;
  finput[629]=0.996078431372549;
  finput[630]=0.996078431372549;
  finput[631]=0.9254901960784314;
  finput[632]=0.0;
  finput[633]=0.0;
  finput[634]=0.0;
  finput[635]=0.0;
  finput[636]=0.0;
  finput[637]=0.0;
  finput[638]=0.0;
  finput[639]=0.0;
  finput[640]=0.0;
  finput[641]=0.0;
  finput[642]=0.0;
  finput[643]=0.0;
  finput[644]=0.0;
  finput[645]=0.0;
  finput[646]=0.0;
  finput[647]=0.0;
  finput[648]=0.0;
  finput[649]=0.0;
  finput[650]=0.0;
  finput[651]=0.0;
  finput[652]=0.027450980392156862;
  finput[653]=0.4588235294117647;
  finput[654]=0.4588235294117647;
  finput[655]=0.6470588235294118;
  finput[656]=0.996078431372549;
  finput[657]=0.996078431372549;
  finput[658]=0.9372549019607843;
  finput[659]=0.19607843137254902;
  finput[660]=0.0;
  finput[661]=0.0;
  finput[662]=0.0;
  finput[663]=0.0;
  finput[664]=0.0;
  finput[665]=0.0;
  finput[666]=0.0;
  finput[667]=0.0;
  finput[668]=0.0;
  finput[669]=0.0;
  finput[670]=0.0;
  finput[671]=0.0;
  finput[672]=0.0;
  finput[673]=0.0;
  finput[674]=0.0;
  finput[675]=0.0;
  finput[676]=0.0;
  finput[677]=0.0;
  finput[678]=0.0;
  finput[679]=0.0;
  finput[680]=0.0;
  finput[681]=0.0;
  finput[682]=0.0;
  finput[683]=0.0;
  finput[684]=0.0;
  finput[685]=0.0;
  finput[686]=0.0;
  finput[687]=0.0;
  finput[688]=0.0;
  finput[689]=0.0;
  finput[690]=0.0;
  finput[691]=0.0;
  finput[692]=0.0;
  finput[693]=0.0;
  finput[694]=0.0;
  finput[695]=0.0;
  finput[696]=0.0;
  finput[697]=0.0;
  finput[698]=0.0;
  finput[699]=0.0;
  finput[700]=0.0;
  finput[701]=0.0;
  finput[702]=0.0;
  finput[703]=0.0;
  finput[704]=0.0;
  finput[705]=0.0;
  finput[706]=0.0;
  finput[707]=0.0;
  finput[708]=0.0;
  finput[709]=0.0;
  finput[710]=0.0;
  finput[711]=0.0;
  finput[712]=0.0;
  finput[713]=0.0;
  finput[714]=0.0;
  finput[715]=0.0;
  finput[716]=0.0;
  finput[717]=0.0;
  finput[718]=0.0;
  finput[719]=0.0;
  finput[720]=0.0;
  finput[721]=0.0;
  finput[722]=0.0;
  finput[723]=0.0;
  finput[724]=0.0;
  finput[725]=0.0;
  finput[726]=0.0;
  finput[727]=0.0;
  finput[728]=0.0;
  finput[729]=0.0;
  finput[730]=0.0;
  finput[731]=0.0;
  finput[732]=0.0;
  finput[733]=0.0;
  finput[734]=0.0;
  finput[735]=0.0;
  finput[736]=0.0;
  finput[737]=0.0;
  finput[738]=0.0;
  finput[739]=0.0;
  finput[740]=0.0;
  finput[741]=0.0;
  finput[742]=0.0;
  finput[743]=0.0;
  finput[744]=0.0;
  finput[745]=0.0;
  finput[746]=0.0;
  finput[747]=0.0;
  finput[748]=0.0;
  finput[749]=0.0;
  finput[750]=0.0;
  finput[751]=0.0;
  finput[752]=0.0;
  finput[753]=0.0;
  finput[754]=0.0;
  finput[755]=0.0;
  finput[756]=0.0;
  finput[757]=0.0;
  finput[758]=0.0;
  finput[759]=0.0;
  finput[760]=0.0;
  finput[761]=0.0;
  finput[762]=0.0;
  finput[763]=0.0;
  finput[764]=0.0;
  finput[765]=0.0;
  finput[766]=0.0;
  finput[767]=0.0;
  finput[768]=0.0;
  finput[769]=0.0;
  finput[770]=0.0;
  finput[771]=0.0;
  finput[772]=0.0;
  finput[773]=0.0;
  finput[774]=0.0;
  finput[775]=0.0;
  finput[776]=0.0;
  finput[777]=0.0;
  finput[778]=0.0;
  finput[779]=0.0;
  finput[780]=0.0;
  finput[781]=0.0;
  finput[782]=0.0;
  finput[783]=0.0;
  assert(3==network(finput));
}