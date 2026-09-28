// Recovery export for failed 2018 Sentinel-2 and mask tasks.
// Cause of the original failures: some exact dates are absent from
// COPERNICUS/S2_SR_HARMONIZED, so qualityMosaic() returned a zero-band image.
//
// This version prefers Level-2A SR. When it is unavailable on the fixed date,
// it falls back to harmonized Level-1C TOA and creates a synthetic SCL band:
// 4 = valid clear pixel, 9 = invalid cloud/cirrus pixel.
// Existing successful Landsat exports do not need to be repeated.

var records = [
  {l8_date:'2018-01-28', s2_date:'2018-01-27'},
  {l8_date:'2018-02-13', s2_date:'2018-02-11'},
  {l8_date:'2018-03-01', s2_date:'2018-03-03'},
  {l8_date:'2018-04-02', s2_date:'2018-04-02'},
  {l8_date:'2018-05-20', s2_date:'2018-05-22'},
  {l8_date:'2018-11-28', s2_date:'2018-11-28'}
];

var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var proj48 = ee.Projection('EPSG:32648');
var l8Transform = [30, 0, 277935, 0, -30, 2727015];
var s2Transform = [10, 0, 277960, 0, -10, 2726990];
var l8Box = ee.Geometry.Rectangle([277935,2694165,294195,2727015], proj48, false);
var s2Box = ee.Geometry.Rectangle([277960,2694180,294190,2726990], proj48, false);

var water2019 = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry:searchBox, scale:30, geometryType:'polygon', eightConnected:true,
  maxPixels:1e7, geometryInNativeProjection:false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000,30);
var shore = lake.buffer(300,30).difference(lake.buffer(-300,30),30);

function l8Valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0)).and(im.select('SR_B2').mask())
    .rename('l8_valid').unmask(0);
}
function prepareSR(im) {
  var sc=im.select('SCL');
  var valid=sc.neq(0).and(sc.neq(1)).and(sc.neq(3)).and(sc.lt(8))
    .and(im.select('B2').mask()).rename('s2_valid').unmask(0);
  return im.select(['B2','B3','B4','B8','B11','B12','SCL'])
    .addBands(valid).copyProperties(im,['system:time_start','CLOUDY_PIXEL_PERCENTAGE','PRODUCT_ID'])
    .set('source_level','L2A_SR');
}
function prepareTOA(im) {
  var qa=im.select('QA60');
  var valid=qa.bitwiseAnd(1 << 10).eq(0)
    .and(qa.bitwiseAnd(1 << 11).eq(0))
    .and(im.select('B2').mask()).rename('s2_valid').unmask(0);
  // Synthetic SCL keeps the exported band schema identical to the SR export.
  var scl=ee.Image(9).where(valid.eq(1),4).rename('SCL').toUint16();
  return im.select(['B2','B3','B4','B8','B11','B12'])
    .addBands(scl).addBands(valid).copyProperties(im,['system:time_start','CLOUDY_PIXEL_PERCENTAGE','PRODUCT_ID'])
    .set('source_level','L1C_TOA_FALLBACK');
}
function getS2(rec) {
  var target=ee.Date(rec.s2_date);
  var start=target.advance(-3,'day'), end=target.advance(4,'day');
  // Query both current harmonized collections and legacy collections because
  // the historical 2018 backfill is not identical across the four catalogs.
  var srH=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi).filterDate(start,end).map(prepareSR);
  var srLegacy=ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(roi).filterDate(start,end).map(prepareSR);
  var toaH=ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
    .filterBounds(roi).filterDate(start,end).map(prepareTOA);
  var toaLegacy=ee.ImageCollection('COPERNICUS/S2')
    .filterBounds(roi).filterDate(start,end).map(prepareTOA);
  var sr=srH.merge(srLegacy);
  var toa=toaH.merge(toaLegacy);
  var candidates=ee.ImageCollection(ee.Algorithms.If(sr.size().gt(0),sr,toa));
  // Select one actual acquisition day first; only tiles from that same day
  // are mosaicked, so the reference never mixes different acquisition dates.
  var best=ee.Image(candidates.sort('CLOUDY_PIXEL_PERCENTAGE').first());
  var bestDay=ee.Date(best.get('system:time_start'));
  var dayStart=ee.Date.parse('YYYY-MM-dd',bestDay.format('YYYY-MM-dd'));
  var chosen=candidates.filterDate(dayStart,dayStart.advance(1,'day'));
  return {
    collection:chosen,
    source:ee.String(ee.Algorithms.If(sr.size().gt(0),'L2A_SR','L1C_TOA_FALLBACK')),
    actualDate:bestDay.format('YYYY-MM-dd'),
    srCount:sr.size(), toaCount:toa.size(), chosenCount:chosen.size()
  };
}
function build(rec) {
  var l8start=ee.Date(rec.l8_date);
  var l8source=ee.Image(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi).filterDate(l8start,l8start.advance(1,'day'))
    .filter(ee.Filter.eq('WRS_PATH',129)).filter(ee.Filter.eq('WRS_ROW',43))
    .sort('CLOUD_COVER').first());
  var lv=l8Valid(l8source);
  var selected=getS2(rec), s2col=selected.collection;
  var s2m=s2col.qualityMosaic('s2_valid');
  var sv=s2col.select('s2_valid').max().unmask(0).rename('s2_valid');
  var joint=lv.and(sv).rename('joint_valid');
  var masks=ee.Image(0).byte().paint(shore,1).rename('shore_zone')
    .addBands(lv).addBands(sv).addBands(joint)
    .addBands(ee.Image(0).byte().paint(roi,1).rename('roi_zone'))
    .toByte().unmask(0).clip(l8Box);
  var s2=s2m.select(['B2','B3','B4','B8','B11','B12','SCL','s2_valid'])
    .toUint16().unmask(0).clip(s2Box);
  return {s2:s2,masks:masks,source:selected.source,actualDate:selected.actualDate,
    srCount:selected.srCount,toaCount:selected.toaCount,chosenCount:selected.chosenCount};
}
function exportRecovery(rec) {
  var item=build(rec), l8tag=rec.l8_date.replace(/-/g,''), s2tag=rec.s2_date.replace(/-/g,'');
  Export.image.toDrive({image:item.s2,
    description:'Fuxian_edgeconfirm_recovery_S2_target'+s2tag+'_for_'+l8tag,
    folder:'Fuxian_edge_confirmatory_2018',region:s2Box,
    crs:'EPSG:32648',crsTransform:s2Transform,maxPixels:1e8,
    fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:0}});
  Export.image.toDrive({image:item.masks,
    description:'Fuxian_edgeconfirm_recovery_masks_native_'+l8tag,
    folder:'Fuxian_edge_confirmatory_2018',region:l8Box,
    crs:'EPSG:32648',crsTransform:l8Transform,maxPixels:1e8,
    fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true}});
}

var panel=ui.Panel({style:{width:'440px',padding:'10px'}});
panel.add(ui.Label('2018 S2 + mask recovery',{fontSize:'18px',fontWeight:'bold'}));
panel.add(ui.Label('SR preferred; automatic TOA fallback. Landsat is not re-exported.'));
var selector=ui.Select({items:records.map(function(r){return r.l8_date;}),value:records[0].l8_date});
var status=ui.Label('Select a date to verify the source collection.');
panel.add(selector); panel.add(status);
panel.add(ui.Button('Prepare first 3 dates (6 tasks)',function(){
  records.slice(0,3).forEach(exportRecovery);
  status.setValue('First 6 recovery tasks prepared. Start them in Tasks.');
}));
panel.add(ui.Button('Prepare last 3 dates (6 tasks)',function(){
  records.slice(3,6).forEach(exportRecovery);
  status.setValue('Last 6 recovery tasks prepared. Start them in Tasks.');
}));
ui.root.insert(0,panel); Map.setCenter(102.8889,24.5022,11);
function show(date) {
  var rec=records.filter(function(r){return r.l8_date===date;})[0], item=build(rec);
  Map.layers().reset();
  Map.addLayer(item.s2,{bands:['B4','B3','B2'],min:0,max:3000},
    'S2 '+rec.s2_date+' (SR/TOA fallback)',true);
  Map.addLayer(item.masks.select('joint_valid').not().selfMask().clip(shore),
    {palette:['ff1744']},'Invalid common shore',false);
  print('RECOVERY PAIR',rec,'actual S2 date',item.actualDate,'source',item.source,
    'SR candidates',item.srCount,'TOA candidates',item.toaCount,
    'chosen-day granules',item.chosenCount);
  ee.Dictionary({source:item.source,actual:item.actualDate,count:item.chosenCount}).evaluate(function(info){
    status.setValue('L8 '+rec.l8_date+' | actual S2 '+info.actual+
      ' | '+info.source+' | '+info.count+' granules');
  });
}
selector.onChange(show); show(records[0].l8_date);
