// Replace the unusable 2024-01-29 confirmatory pair with the prediction-blind
// quality-screened Landsat 8 date 2024-04-18. This script creates exactly three
// required GeoTIFF tasks after the button is clicked.

var l8Date = '2024-04-18';
var seed = ee.Geometry.Point([102.8889,24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78,24.28,103.00,24.72],null,false);
var proj48 = ee.Projection('EPSG:32648');
var l8Transform=[30,0,277935,0,-30,2727015];
var s2Transform=[10,0,277960,0,-10,2726990];
var l8Box=ee.Geometry.Rectangle([277935,2694165,294195,2727015],proj48,false);
var s2Box=ee.Geometry.Rectangle([277960,2694180,294190,2726990],proj48,false);
var water2019=ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year',2019)).first()).select('waterClass').gte(2);
var components=water2019.selfMask().toInt().reduceToVectors({
  geometry:searchBox,scale:30,geometryType:'polygon',eightConnected:true,
  maxPixels:1e7,geometryInNativeProjection:false
});
var lake=ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi=lake.buffer(1000,30);
var shore=lake.buffer(300,30).difference(lake.buffer(-300,30),30);

function l8Valid(im){return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
  .and(im.select('QA_RADSAT').eq(0)).and(im.select('SR_B2').mask())
  .rename('l8_valid').unmask(0);}
function s2Valid(im){var sc=im.select('SCL');return sc.neq(0).and(sc.neq(1))
  .and(sc.neq(3)).and(sc.lt(8)).and(im.select('B2').mask())
  .rename('s2_valid').unmask(0);}
function fraction(mask,scale){return ee.Number(mask.rename('clear').reduceRegion({
  reducer:ee.Reducer.mean(),geometry:shore,scale:scale,maxPixels:1e8,tileScale:4
}).get('clear'));}

var start=ee.Date(l8Date);
var source=ee.Image(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi).filterDate(start,start.advance(1,'day'))
  .filter(ee.Filter.eq('WRS_PATH',129)).filter(ee.Filter.eq('WRS_ROW',43))
  .sort('CLOUD_COVER').first());
var offsets=ee.List.sequence(-2,2);
var choices=ee.FeatureCollection(offsets.map(function(o){
  o=ee.Number(o);var day=start.advance(o,'day');
  var col=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi).filterDate(day,day.advance(1,'day'))
    .map(function(im){return im.addBands(s2Valid(im));});
  var exists=col.size().gt(0);
  var valid=ee.Image(ee.Algorithms.If(exists,col.select('s2_valid').max().unmask(0),
    ee.Image(0).rename('s2_valid')));
  var clear=fraction(valid,20);
  return ee.Feature(null,{s2_date:day.format('YYYY-MM-dd'),s2_clear_shore:clear,
    scene_count:col.size(),day_gap:o.abs(),score:clear.subtract(o.abs().multiply(0.025))});
})).filter(ee.Filter.gt('scene_count',0)).sort('score',false);
var choice=ee.Feature(choices.first());
print('REPLACEMENT_PAIR',choice);

var panel=ui.Panel({style:{width:'430px',padding:'10px'}});
panel.add(ui.Label('Replace low-coverage confirmation date',{fontSize:'18px',fontWeight:'bold'}));
var status=ui.Label('Checking the best Sentinel-2 pair for 2024-04-18...');panel.add(status);
ui.root.insert(0,panel);Map.setCenter(102.8889,24.5022,11);

choice.evaluate(function(rec,err){
  if(err){status.setValue('Pair selection failed: '+err);return;}
  status.setValue('L8 2024-04-18 / S2 '+rec.properties.s2_date+
    ' | S2 clear shore '+(100*rec.properties.s2_clear_shore).toFixed(1)+'%');
  var s2Date=rec.properties.s2_date;
  var reflectance=source.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2).rename(['B2','B3','B4','B5','B6','B7']);
  var lv=l8Valid(source);
  var sd=ee.Date(s2Date);
  var s2col=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi).filterDate(sd,sd.advance(1,'day'))
    .map(function(im){return im.addBands(s2Valid(im));});
  var s2m=s2col.qualityMosaic('s2_valid');
  var sv=s2col.select('s2_valid').max().unmask(0).rename('s2_valid');
  var masks=ee.Image(0).byte().paint(shore,1).rename('shore_zone')
    .addBands(lv).addBands(sv).addBands(lv.and(sv).rename('joint_valid'))
    .addBands(ee.Image(0).byte().paint(roi,1).rename('roi_zone')).toByte().unmask(0).clip(l8Box);
  var l8=reflectance.addBands(lv.rename('valid')).toFloat().unmask(-9999).clip(l8Box);
  var s2=s2m.select(['B2','B3','B4','B8','B11','B12','SCL','s2_valid'])
    .toUint16().unmask(0).clip(s2Box);
  var s2tag=s2Date.replace(/-/g,'');
  panel.add(ui.Button('Prepare replacement: 3 GeoTIFF tasks',function(){
    Export.image.toDrive({image:l8,description:'Fuxian_edgeconfirm_post2019_L8_native_20240418',
      folder:'Fuxian_edge_confirmatory_post2019',region:l8Box,crs:'EPSG:32648',
      crsTransform:l8Transform,maxPixels:1e8,fileFormat:'GeoTIFF',
      formatOptions:{cloudOptimized:true,noData:-9999}});
    Export.image.toDrive({image:s2,
      description:'Fuxian_edgeconfirm_post2019_S2_'+s2tag+'_for_20240418',
      folder:'Fuxian_edge_confirmatory_post2019',region:s2Box,crs:'EPSG:32648',
      crsTransform:s2Transform,maxPixels:1e8,fileFormat:'GeoTIFF',
      formatOptions:{cloudOptimized:true,noData:0}});
    Export.image.toDrive({image:masks,
      description:'Fuxian_edgeconfirm_post2019_masks_native_20240418',
      folder:'Fuxian_edge_confirmatory_post2019',region:l8Box,crs:'EPSG:32648',
      crsTransform:l8Transform,maxPixels:1e8,fileFormat:'GeoTIFF',
      formatOptions:{cloudOptimized:true}});
    status.setValue('Three replacement tasks prepared. Start them in Tasks.');
  }));
  Map.layers().reset();
  Map.addLayer(l8,{bands:['B4','B3','B2'],min:0,max:0.25},'L8 2024-04-18',true);
  Map.addLayer(s2,{bands:['B4','B3','B2'],min:0,max:3000},'S2 '+s2Date,false);
  Map.addLayer(masks.select('joint_valid').not().selfMask().clip(shore),
    {palette:['ff1744']},'Invalid common shore',false);
});
