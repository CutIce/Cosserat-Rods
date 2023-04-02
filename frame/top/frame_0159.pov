#include "E:/Documents/23SP/graduation_project/PyElastica-master/examples/Visualization/default.inc"

camera{
    location <0,15,3>
    angle 30
    look_at <0.0,0,3>
    sky <-1,0,0>
    right x*image_width/image_height
}
light_source{
    <15,10.5,-15>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<0.0,0.0,0.0>,0.08
    ,<0.00013252476173699606,0.0,0.10052940550406025>,0.08
    ,<0.0002650318636142427,0.0,0.20103899337728226>,0.08
    ,<0.0003975154156098837,0.0,0.30152877143467843>,0.08
    ,<0.0005299765704485835,0.0,0.4019987474833247>,0.08
    ,<0.0006624166651575126,0.0,0.5024489293255987>,0.08
    ,<0.0007948369998611533,0.0,0.6028793247592884>,0.08
    ,<0.00092723859782862,0.0,0.7032899415776948>,0.08
    ,<0.0010596220084980042,0.0,0.8036807875697347>,0.08
    ,<0.001191987206470228,0.0,0.9040518705200109>,0.08
    ,<0.001324333622203077,0.0,1.0044031982087673>,0.08
    ,<0.0014566603066186801,0.0,1.1047347784118475>,0.08
    ,<0.001588966218329411,0.0,1.2050466189005415>,0.08
    ,<0.0017212505801358167,0.0,1.305338727441418>,0.08
    ,<0.0018535132378008012,0.0,1.405611111796188>,0.08
    ,<0.001985754945313322,0.0,1.5058637797215437>,0.08
    ,<0.0021179775067581745,0.0,1.6060967389691216>,0.08
    ,<0.0022501837303624808,0.0,1.7063099972854907>,0.08
    ,<0.002382377167641291,0.0,1.8065035624123236>,0.08
    ,<0.002514561664127571,0.0,1.906677442086574>,0.08
    ,<0.0026467609940837127,0.0,2.0068316440312404>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0026467380445224457,0.0,2.00687693252555>,0.05
    ,<0.002646567974634046,0.0,2.0820926847829364>,0.05
    ,<0.0026464066500530973,0.0,2.157297342630281>,0.05
    ,<0.0026462540681831516,0.0,2.2324909093399716>,0.05
    ,<0.002646110226434986,0.0,2.3076733881829505>,0.05
    ,<0.002645975122258567,0.0,2.3828447824287213>,0.05
    ,<0.0026458487531031342,0.0,2.458005095345335>,0.05
    ,<0.002645731116385805,0.0,2.5331543301994026>,0.05
    ,<0.0026456222095244766,0.0,2.608292490256087>,0.05
    ,<0.0026455220299569917,0.0,2.6834195787791035>,0.05
    ,<0.002645430575125318,0.0,2.7585355990307425>,0.05
    ,<0.0026453478424617134,0.0,2.8336405542718324>,0.05
    ,<0.0026452738293903724,0.0,2.90873444776177>,0.05
    ,<0.002645208533344016,0.0,2.9838172827585105>,0.05
    ,<0.002645151951764248,0.0,3.0588890625185665>,0.05
    ,<0.002645104082087294,0.0,3.133949790297025>,0.05
    ,<0.0026450649217445903,0.0,3.208999469347535>,0.05
    ,<0.002645034468172023,0.0,3.284038102922283>,0.05
    ,<0.0026450127188134857,0.0,3.35906569427204>,0.05
    ,<0.0026449996711171877,0.0,3.4340822466461542>,0.05
    ,<0.002644995322525409,0.0,3.50908776329251>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }