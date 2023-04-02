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
    ,<0.0001074691564874541,0.0,0.10052940534577372>,0.08
    ,<0.00021492268607040685,0.0,0.2010389930684655>,0.08
    ,<0.00032235739601682383,0.0,0.3015287709814409>,0.08
    ,<0.00042977436193900054,0.0,0.40199874689184123>,0.08
    ,<0.0005371748165488752,0.0,0.5024489286021345>,0.08
    ,<0.0006445599735913675,0.0,0.6028793239101621>,0.08
    ,<0.0007519308332942622,0.0,0.7032899406092609>,0.08
    ,<0.0008592880193209878,0.0,0.8036807864883078>,0.08
    ,<0.0009666316908904525,0.0,0.9040518693317292>,0.08
    ,<0.0010739615600618,0.0,1.0044031969195475>,0.08
    ,<0.0011812770177540288,0.0,1.1047347770273217>,0.08
    ,<0.0012885773600148657,0.0,1.2050466174260324>,0.08
    ,<0.001395862074835211,0.0,1.305338725882021>,0.08
    ,<0.0015031311347294062,0.0,1.405611110156854>,0.08
    ,<0.001610385234351686,0.0,1.5058637780072395>,0.08
    ,<0.0017176259153664731,0.0,1.6060967371849921>,0.08
    ,<0.0018248555424140447,0.0,1.7063099954370355>,0.08
    ,<0.0019320771043436688,0.0,1.8065035605055044>,0.08
    ,<0.002039293859406263,0.0,1.9066774401278286>,0.08
    ,<0.0021465243603086544,0.0,2.0068316420310697>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.002146501410802424,0.0,2.006876930525388>,0.05
    ,<0.0021463313413136427,0.0,2.0820926827828226>,0.05
    ,<0.0021461700171086042,0.0,2.1572973406302034>,0.05
    ,<0.0021460174356193124,0.0,2.2324909073399333>,0.05
    ,<0.0021458735942621915,0.0,2.3076733861829597>,0.05
    ,<0.0021457384904563044,0.0,2.382844780428768>,0.05
    ,<0.002145612121628112,0.0,2.458005093345414>,0.05
    ,<0.002145494485208887,0.0,2.533154328199531>,0.05
    ,<0.0021453855786338454,0.0,2.608292488256248>,0.05
    ,<0.0021452853993323352,0.0,2.683419576779294>,0.05
    ,<0.0021451939447298075,0.0,2.758535597030951>,0.05
    ,<0.0021451112122607245,0.0,2.833640552272078>,0.05
    ,<0.002145037199366091,0.0,2.908734445762024>,0.05
    ,<0.0021449719034818734,0.0,2.98381728075879>,0.05
    ,<0.002144915322041763,0.0,3.0588890605188768>,0.05
    ,<0.002144867452481647,0.0,3.133949788297353>,0.05
    ,<0.0021448282922399333,0.0,3.208999467347863>,0.05
    ,<0.0021447978387587265,0.0,3.2840381009226207>,0.05
    ,<0.0021447760894772318,0.0,3.3590656922723863>,0.05
    ,<0.002144763041831599,0.0,3.434082244646502>,0.05
    ,<0.002144758693257245,0.0,3.509087761292864>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }