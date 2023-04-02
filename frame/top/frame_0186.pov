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
    ,<0.0001550746826335827,0.0,0.10052940567472628>,0.08
    ,<0.00031012604156091517,0.0,0.2010389937122465>,0.08
    ,<0.0004651492240490067,0.0,0.30152877192754257>,0.08
    ,<0.0006201452624917367,0.0,0.4019987481277335>,0.08
    ,<0.0007751155897421413,0.0,0.5024489301151156>,0.08
    ,<0.0009300618804044566,0.0,0.6028793256872286>,0.08
    ,<0.001084985791693419,0.0,0.7032899426370126>,0.08
    ,<0.0012398886656280795,0.0,0.8036807887529824>,0.08
    ,<0.001394771263548077,0.0,0.9040518718193711>,0.08
    ,<0.0015496336003001144,0.0,1.0044031996162166>,0.08
    ,<0.0017044749298531948,0.0,1.104734779919395>,0.08
    ,<0.0018592939062579124,0.0,1.2050466205005126>,0.08
    ,<0.0020140889005247136,0.0,1.3053387291267526>,0.08
    ,<0.002168858437379949,0.0,1.4056111135606228>,0.08
    ,<0.0023236016674777664,0.0,1.5058637815597176>,0.08
    ,<0.0024783187838478557,0.0,1.606096740876465>,0.08
    ,<0.0026330112948977666,0.0,1.7063099992580082>,0.08
    ,<0.0027876820700521313,0.0,1.8065035644461678>,0.08
    ,<0.002942335121450639,0.0,1.9066774441775793>,0.08
    ,<0.0030969738933093707,0.0,2.0068316461846205>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.003096950937769679,0.0,2.0068769346789073>,0.05
    ,<0.003096780823567844,0.0,2.082092686936236>,0.05
    ,<0.003096619456934868,0.0,2.1572973447835158>,0.05
    ,<0.003096466835286545,0.0,2.2324909114931413>,0.05
    ,<0.003096322956042497,0.0,2.307673390336063>,0.05
    ,<0.0030961878166432246,0.0,2.3828447845817844>,0.05
    ,<0.003096061414524024,0.0,2.458005097498348>,0.05
    ,<0.0030959437471030247,0.0,2.533154332352361>,0.05
    ,<0.003095834811808717,0.0,2.6082924924090003>,0.05
    ,<0.00309573460608664,0.0,2.6834195809319823>,0.05
    ,<0.0030956431273759678,0.0,2.758535601183581>,0.05
    ,<0.0030955603731066877,0.0,2.8336405564246356>,0.05
    ,<0.003095486340706464,0.0,2.90873444991454>,0.05
    ,<0.0030954210276113547,0.0,2.983817284911256>,0.05
    ,<0.0030953644312599403,0.0,3.0588890646712943>,0.05
    ,<0.0030953165490832544,0.0,3.1339497924497413>,0.05
    ,<0.003095277378517205,0.0,3.2089994715002264>,0.05
    ,<0.0030952469170046674,0.0,3.2840381050749636>,0.05
    ,<0.003095225161984944,0.0,3.359065696424718>,0.05
    ,<0.0030952121108957995,0.0,3.434082248798828>,0.05
    ,<0.003095207761173905,0.0,3.5090877654451864>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }