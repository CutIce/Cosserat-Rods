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
    ,<2.6452468456487035e-05,0.0,0.10053125690816443>,0.08
    ,<5.289961166782145e-05,0.0,0.2010426876109997>,0.08
    ,<7.934304833629826e-05,0.0,0.3015342879803924>,0.08
    ,<0.00010578350522523552,0.0,0.4020060641207014>,0.08
    ,<0.0001322218008236213,0.0,0.502458015830843>,0.08
    ,<0.0001586587972864342,0.0,0.6028901399614675>,0.08
    ,<0.0001850954779055397,0.0,0.7033024331604534>,0.08
    ,<0.00021153291071951562,0.0,0.8036948964323368>,0.08
    ,<0.00023797213821327292,0.0,0.90406753644003>,0.08
    ,<0.0002644141600253927,0.0,1.004420364541617>,0.08
    ,<0.0002908598395592118,0.0,1.1047533891782764>,0.08
    ,<0.00031730985683174444,0.0,1.205066609988211>,0.08
    ,<0.00034376475307020384,0.0,1.3053600471874593>,0.08
    ,<0.00037022493619224785,0.0,1.4056337173864302>,0.08
    ,<0.00039669071614647545,0.0,1.5058876163569468>,0.08
    ,<0.0004231624136949978,0.0,1.606121751186758>,0.08
    ,<0.00044964043232126083,0.0,1.706336143996983>,0.08
    ,<0.000476125294389421,0.0,1.8065308091651688>,0.08
    ,<0.0005026177803069273,0.0,1.906705756603341>,0.08
    ,<0.0005291126385712192,0.0,2.0068609903359773>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.000529089339834709,0.0,2.0069065962512>,0.05
    ,<0.0005289166611085435,0.0,2.082123893188883>,0.05
    ,<0.0005287528057551539,0.0,2.157330068658233>,0.05
    ,<0.0005285978103371776,0.0,2.232525117296239>,0.05
    ,<0.000528451710223713,0.0,2.307709032659032>,0.05
    ,<0.0005283145198737016,0.0,2.3828818076075113>,0.05
    ,<0.0005281862444336664,0.0,2.4580434322153657>,0.05
    ,<0.0005280668587747381,0.0,2.5331938968043324>,0.05
    ,<0.000527956341133581,0.0,2.6083331999977535>,0.05
    ,<0.0005278546726961892,0.0,2.683461349691458>,0.05
    ,<0.0005277618317042636,0.0,2.758578352604773>,0.05
    ,<0.0005276778149117962,0.0,2.8336842059830447>,0.05
    ,<0.0005276026391231176,0.0,2.9087789028686486>,0.05
    ,<0.0005275362934223278,0.0,2.983862442298295>,0.05
    ,<0.0005274787622126674,0.0,3.05893482989958>,0.05
    ,<0.0005274300761381776,0.0,3.1339960704079624>,0.05
    ,<0.0005273902472415661,0.0,3.2090461645650468>,0.05
    ,<0.0005273592517741783,0.0,3.2840851137593785>,0.05
    ,<0.000527337087250019,0.0,3.3591129241576674>,0.05
    ,<0.0005273237916866845,0.0,3.4341296048608076>,0.05
    ,<0.0005273193683095571,0.0,3.509135163753087>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }