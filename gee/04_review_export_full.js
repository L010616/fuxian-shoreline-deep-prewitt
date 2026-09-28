// DEPRECATED FOR EXPORT: zero-origin L8 grid risks half-pixel shifts.
// Use 05_native_grid_reexport.js for the first-scene repair.
// Fuxian Lake pilot: inventory and pre-2020 screening geometry.
// The 2019 JRC outline is a screening prior, NEVER a reference label.
// No full-period maximum extent is used. No export is started automatically.
var cfg = {start: '2020-01-01', end: '2026-01-01', year: 2020};
var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var water2019 = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox, scale: 30, geometryType: 'polygon',
  eightConnected: true, maxPixels: 1e7, geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(seed).filterDate(cfg.start, cfg.end);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi).filterDate(cfg.start, cfg.end);
function l8valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0))
    .and(im.select('SR_B2').mask()).rename('valid').unmask(0);
}

var pilotRows=[{"calendar_gap_days":0,"joint_shore_valid":0.9664507278308317,"l8_date":"2020-01-02","l8_id":"LC08_129043_20200102","l8_shore_valid":0.9723813343723652,"s2_date":"2020-01-02","s2_granule_ids":["20200102T034141_20200102T034813_T47QRG","20200102T034141_20200102T034813_T47RRH","20200102T034141_20200102T034813_T48QTM","20200102T034141_20200102T034813_T48RTN"],"s2_shore_valid":0.9912722837095312,"signed_time_gap_hours":0.2686763888888889,"year":2020,"split":"train","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":2,"joint_shore_valid":0.9985756067890064,"l8_date":"2020-05-09","l8_id":"LC08_129043_20200509","l8_shore_valid":0.9994321662856203,"s2_date":"2020-05-11","s2_granule_ids":["20200511T033541_20200511T033837_T47QRG","20200511T033541_20200511T033837_T47RRH","20200511T033541_20200511T033837_T48QTM","20200511T033541_20200511T033837_T48RTN"],"s2_shore_valid":0.9991573974904925,"signed_time_gap_hours":48.28520638888889,"year":2020,"split":"train","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":0,"joint_shore_valid":0.9999160688906545,"l8_date":"2021-02-05","l8_id":"LC08_129043_20210205","l8_shore_valid":0.9999328552760645,"s2_date":"2021-02-05","s2_granule_ids":["20210205T033931_20210205T034427_T47QRG","20210205T033931_20210205T034427_T47RRH","20210205T033931_20210205T034427_T48QTM","20210205T033931_20210205T034427_T48RTN"],"s2_shore_valid":0.9999832137781309,"signed_time_gap_hours":0.2716511111111111,"year":2021,"split":"train","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":1,"joint_shore_valid":0.8660870263570022,"l8_date":"2021-05-12","l8_id":"LC08_129043_20210512","l8_shore_valid":0.994737828488282,"s2_date":"2021-05-11","s2_granule_ids":["20210511T033539_20210511T034807_T47QRG","20210511T033539_20210511T034807_T47RRH","20210511T033539_20210511T034807_T48QTM","20210511T033539_20210511T034807_T48RTN"],"s2_shore_valid":0.8717291717546479,"signed_time_gap_hours":-23.719184444444444,"year":2021,"split":"train","qa_group":"partial","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":0,"joint_shore_valid":0.9999162005472967,"l8_date":"2022-03-12","l8_id":"LC08_129043_20220312","l8_shore_valid":0.9999328552760645,"s2_date":"2022-03-12","s2_granule_ids":["20220312T033541_20220312T034214_T47QRG","20220312T033541_20220312T034214_T47RRH","20220312T033541_20220312T034214_T48QTM","20220312T033541_20220312T034214_T48RTN"],"s2_shore_valid":0.9999832137781309,"signed_time_gap_hours":0.2755563888888889,"year":2022,"split":"train","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":0,"joint_shore_valid":0.9690618081854887,"l8_date":"2022-08-19","l8_id":"LC08_129043_20220819","l8_shore_valid":0.973686246002338,"s2_date":"2022-08-19","s2_granule_ids":["20220819T033551_20220819T034355_T47QRG","20220819T033551_20220819T034355_T47RRH","20220819T033551_20220819T034355_T48QTM","20220819T033551_20220819T034355_T48RTN"],"s2_shore_valid":0.9874624038001374,"signed_time_gap_hours":0.2683775,"year":2022,"split":"train","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":0,"joint_shore_valid":0.9999328551125236,"l8_date":"2023-01-26","l8_id":"LC08_129043_20230126","l8_shore_valid":0.9999328552760646,"s2_date":"2023-01-26","s2_granule_ids":["20230126T034021_20230126T034302_T47QRG","20230126T034021_20230126T034302_T47RRH","20230126T034021_20230126T034302_T48QTM","20230126T034021_20230126T034302_T48RTN"],"s2_shore_valid":1,"signed_time_gap_hours":0.26682,"year":2023,"split":"validation","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":1,"joint_shore_valid":0.8982395532100194,"l8_date":"2023-03-31","l8_id":"LC08_129043_20230331","l8_shore_valid":0.8982366413103176,"s2_date":"2023-04-01","s2_granule_ids":["20230401T033539_20230401T034555_T47QRG","20230401T033539_20230401T034555_T47RRH","20230401T033539_20230401T034555_T48QTM","20230401T033539_20230401T034555_T48RTN"],"s2_shore_valid":0.9993749600915803,"signed_time_gap_hours":24.27824777777778,"year":2023,"split":"validation","qa_group":"partial","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":1,"joint_shore_valid":0.9374735617005566,"l8_date":"2024-02-14","l8_id":"LC08_129043_20240214","l8_shore_valid":0.9982878095396458,"s2_date":"2024-02-15","s2_granule_ids":["20240215T033839_20240215T034800_T47QRG","20240215T033839_20240215T034800_T47RRH","20240215T033839_20240215T034800_T48QTM","20240215T033839_20240215T034800_T48RTN"],"s2_shore_valid":0.9391857563312036,"signed_time_gap_hours":24.272789444444445,"year":2024,"split":"test","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":1,"joint_shore_valid":0.9172807242168575,"l8_date":"2024-05-04","l8_id":"LC08_129043_20240504","l8_shore_valid":0.9820598513586963,"s2_date":"2024-05-05","s2_granule_ids":["20240505T033539_20240505T034925_T47QRG","20240505T033539_20240505T034925_T47RRH","20240505T033539_20240505T034925_T48QTM","20240505T033539_20240505T034925_T48RTN"],"s2_shore_valid":0.935918235958984,"signed_time_gap_hours":24.285254444444444,"year":2024,"split":"test","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":2,"joint_shore_valid":0.951835063191897,"l8_date":"2025-05-07","l8_id":"LC08_129043_20250507","l8_shore_valid":0.9995264980399335,"s2_date":"2025-05-05","s2_granule_ids":["20250505T033601_20250505T035053_T47QRG","20250505T033601_20250505T035053_T47RRH","20250505T033601_20250505T035053_T48QTM","20250505T033601_20250505T035053_T48RTN"],"s2_shore_valid":0.9522312180280076,"signed_time_gap_hours":-47.710214722222226,"year":2025,"split":"test","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":0,"joint_shore_valid":0.9972710869506545,"l8_date":"2025-12-01","l8_id":"LC08_129043_20251201","l8_shore_valid":0.9978345826530814,"s2_date":"2025-12-01","s2_granule_ids":["20251201T034131_20251201T034125_T47QRG","20251201T034131_20251201T034125_T47RRH","20251201T034131_20251201T034125_T48QTM","20251201T034131_20251201T034125_T48RTN"],"s2_shore_valid":0.9994243313323717,"signed_time_gap_hours":0.27161972222222225,"year":2025,"split":"test","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false}];
// Append to the geometry/data header, with pilotRows defined from the paired manifest.
// Data exports and water-index layers are not reference labels.
var exportBox=roi.bounds(30,ee.Projection('EPSG:32648'));
var grid30=[30,0,0,0,-30,0], grid10=[10,0,0,0,-10,0];
var panel=ui.Panel({style:{width:'370px',padding:'8px'}});
panel.add(ui.Label('Fuxian pilot: image review',{fontSize:'18px',fontWeight:'bold'}));
panel.add(ui.Label('QA passed; temporal consistency, registration and manual labels still pending.'));
var info=ui.Label('');panel.add(info);
var select=ui.Select({items:pilotRows.map(function(r){return r.l8_date;}),value:pilotRows[0].l8_date});panel.add(select);
var state=ui.Label('');panel.add(state);
var direct=ui.Label('Selected L8 download link will appear here');panel.add(direct);
var metadataBox=ui.Textbox({placeholder:'SELECTED_REVIEW_METADATA',style:{stretch:'horizontal'}});panel.add(metadataBox);
var geometryBox=ui.Textbox({placeholder:'SCREENING_GEOMETRY_JSON',style:{stretch:'horizontal'}});panel.add(geometryBox);
ui.root.insert(0,panel);Map.setCenter(102.8889,24.5022,11);
var selected;
function s2valid(im){
  var sc=im.select('SCL');
  return sc.neq(0).and(sc.neq(1)).and(sc.neq(3)).and(sc.lt(8))
    .and(im.select('B2').mask()).rename('s2_valid').unmask(0);
}
function showDate(date){
  var rec=pilotRows.filter(function(r){return r.l8_date===date;})[0];
  var lim=ee.Image('LANDSAT/LC08/C02/T1_L2/'+rec.l8_id);
  var refl=lim.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2).rename(['B2','B3','B4','B5','B6','B7']);
  var lv=l8valid(lim);
  var sims=ee.ImageCollection(rec.s2_granule_ids.map(function(id){
    var im=ee.Image('COPERNICUS/S2_SR_HARMONIZED/'+id);
    return im.addBands(s2valid(im));
  }));
  var sm=sims.qualityMosaic('s2_valid'), sv=sims.select('s2_valid').max().unmask(0);
  var joint=lv.and(sv).rename('joint_valid');
  var shoreMask=ee.Image(0).byte().paint(shore,1).rename('shore_zone');
  var mndwi=refl.expression('(g-s)/(g+s)',{g:refl.select('B3'),s:refl.select('B6')}).rename('MNDWI');
  var awei=refl.expression('b+2.5*g-1.5*(n+s)-0.25*t',{
    b:refl.select('B2'),g:refl.select('B3'),n:refl.select('B5'),s:refl.select('B6'),t:refl.select('B7')});
  selected={rec:rec,l8:refl.addBands(lv).toFloat().unmask(-9999).clip(exportBox),
    s2:sm.select(['B2','B3','B4','B8','B11','B12','SCL','s2_valid']).toUint16().unmask(0).clip(exportBox),
    masks:shoreMask.addBands(lv.rename('l8_valid')).addBands(sv)
      .addBands(joint).toByte().unmask(0).clip(exportBox)};
  Map.layers().reset();
  Map.addLayer(refl.clip(roi),{bands:['B4','B3','B2'],min:0,max:.3},'L8 RGB '+rec.l8_date,true);
  Map.addLayer(sm.clip(roi),{bands:['B4','B3','B2'],min:0,max:3000},'S2 RGB '+rec.s2_date,false);
  Map.addLayer(sm.clip(roi),{bands:['B8','B4','B3'],min:0,max:4000},'S2 NIR false color',false);
  Map.addLayer(mndwi.updateMask(lv).clip(roi),{min:-.5,max:.5,palette:['a6611a','f5f5f5','018571']},'MNDWI diagnostic',false);
  Map.addLayer(awei.gt(0).updateMask(lv).selfMask().clip(roi),{palette:['42a5f5']},'AWEI > 0 prediction (not label)',false);
  Map.addLayer(joint.not().selfMask().clip(shore),{palette:['ff1744']},'Invalid common shore pixels',false);
  Map.addLayer(ee.Image().byte().paint(lake,1,1),{palette:['ffff00']},'JRC 2019 screening outline',false);
  info.setValue(rec.split+' | L8 '+rec.l8_date+' | S2 '+rec.s2_date+
    ' | common QA '+(100*rec.joint_shore_valid).toFixed(2)+'%');
  metadataBox.setValue(JSON.stringify(rec));
  state.setValue('Toggle image layers to review. No manual labels exist yet.');
  direct.setValue('Selected L8 download link will appear here');direct.setUrl(null);
}
select.onChange(showDate);showDate(pilotRows[0].l8_date);
panel.add(ui.Button('Prepare selected-date Drive exports',function(){
  throw new Error('Old zero-origin export disabled. Use 05_native_grid_reexport.js.');
  var tag=selected.rec.l8_date.replace(/-/g,'');
  Export.image.toDrive({image:selected.l8,description:'Fuxian_L8_'+tag,
    folder:'Fuxian_pilot',region:exportBox,crs:'EPSG:32648',crsTransform:grid30,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:-9999}});
  Export.image.toDrive({image:selected.s2,description:'Fuxian_S2_'+selected.rec.s2_date.replace(/-/g,''),
    folder:'Fuxian_pilot',region:exportBox,crs:'EPSG:32648',crsTransform:grid10,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:0}});
  Export.image.toDrive({image:selected.masks,description:'Fuxian_masks_'+tag,
    folder:'Fuxian_pilot',region:exportBox,crs:'EPSG:32648',crsTransform:grid30,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true}});
  state.setValue('Three export tasks prepared in Tasks. S2 B11/B12/SCL are native 20m resampled to a 10m grid.');
}));
panel.add(ui.Button('Create direct L8 GeoTIFF link',function(){
  throw new Error('Old zero-origin export disabled. Use 05_native_grid_reexport.js.');
  state.setValue('Preparing direct L8 download...');
  selected.l8.getDownloadURL({name:'Fuxian_L8_'+selected.rec.l8_date.replace(/-/g,''),
    region:exportBox,crs:'EPSG:32648',crs_transform:grid30,format:'GEO_TIFF',filePerBand:false},
    function(url,error){if(error){state.setValue('Download error: '+error);return;}
      direct.setValue('Download selected L8 GeoTIFF');direct.setUrl(url);
      state.setValue('L8 download ready. It contains B2-B7 reflectance and valid mask.');});
}));
ee.FeatureCollection([ee.Feature(lake,{role:'screening_outline_2019_not_truth'}),
  ee.Feature(shore,{role:'shore_zone_300m'}),ee.Feature(exportBox,{role:'export_bounds'})])
  .evaluate(function(result,error){if(!error){geometryBox.setValue(JSON.stringify(result));}});
