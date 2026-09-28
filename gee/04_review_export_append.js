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
