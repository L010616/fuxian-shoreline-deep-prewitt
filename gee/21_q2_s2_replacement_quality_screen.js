// Q2 external-lake repair: screen ONLY the five failed Sentinel-2 pairings.
// Run in the GEE Code Editor. Pick one target, click Scan alternatives, then
// export S2 + QA/JRC for an eligible replacement date. Reuse existing L8.
// The screening uses only source-image QA and the fixed 2019 JRC water map;
// no model prediction or reference shoreline is read.
// Do not use an alternative if none passes; replace the Landsat date instead.

var CFG = {
  folder: 'Fuxian_Q2_external_replacements',
  gap: 4,
  specs: {
    Dianchi_Waihai: {box: [102.57, 24.62, 102.87, 25.06], crs: 'EPSG:32648'},
    Erhai: {box: [100.04, 25.56, 100.35, 26.02], crs: 'EPSG:32647'},
    Xingyun: {box: [102.69, 24.25, 102.88, 24.43], crs: 'EPSG:32648',
      seed: [102.7820, 24.3370], qualityRadiusM: 5900}
  },
  targets: [
    {lake: 'Dianchi_Waihai', l8: '2020-07-28', rejectedS2: '2020-07-28'},
    {lake: 'Dianchi_Waihai', l8: '2021-11-20', rejectedS2: '2021-11-20'},
    {lake: 'Dianchi_Waihai', l8: '2022-03-12', rejectedS2: '2022-03-10'},
    {lake: 'Dianchi_Waihai', l8: '2024-10-27', rejectedS2: '2024-10-25'},
    {lake: 'Xingyun', l8: '2021-11-20', rejectedS2: '2021-11-17'}
  ]
};

var jrc = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass');

function l8Valid(image) {
  return image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0))
    .and(image.select('SR_B2').mask())
    .rename('l8_valid').unmask(0, false).toByte();
}

function s2Valid(image) {
  var scl = image.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.lt(8)).and(image.select('B2').mask())
    .rename('s2_valid').unmask(0).toByte();
}

function products(target, s2Day) {
  var spec = CFG.specs[target.lake];
  var box = ee.Geometry.Rectangle(spec.box, null, false);
  var source = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(box)
    .filterDate(target.l8, ee.Date(target.l8).advance(1, 'day'))
    .sort('CLOUD_COVER', false).mosaic();
  var lm = l8Valid(source);
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(box)
    .filterDate(s2Day, ee.Date(s2Day).advance(1, 'day'));
  var foot = collection.map(function(image) {
    return image.select('B2').mask().unmask(0, false)
      .rename('s2_foot').toByte();
  }).max().unmask(0, false).rename('s2_foot').toByte();
  var withValid = collection.map(function(image) {
    return image.addBands(s2Valid(image));
  });
  var sm = withValid.select('s2_valid').max()
    .unmask(0, false).rename('s2_valid').toByte();
  var s2 = withValid.qualityMosaic('s2_valid')
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL', 's2_valid'])
    .toUint16().unmask(0).clip(box);
  var joint = lm.and(sm).rename('joint_valid').toByte();
  var masks = jrc.unmask(0).toByte().rename('jrc2019_class')
    .addBands(lm).addBands(sm).addBands(joint)
    .toByte().unmask(0).clip(box);
  return {box: box, spec: spec, foot: foot, lm: lm,
    sm: sm, joint: joint, s2: s2, masks: masks};
}

function qualityCheck(target, s2Day) {
  var p = products(target, s2Day);
  var water = jrc.gte(2).unmask(0, false).toByte();
  // The Xingyun export box also contains a piece of Fuxian Lake. The fixed
  // 5.9 km seed neighbourhood contains the Xingyun JRC component but excludes
  // the neighbouring lake; local QC still uses the exact seed component.
  var spec = CFG.specs[target.lake];
  if (spec.qualityRadiusM) {
    var s = spec.seed;
    var seedBox = ee.Geometry.Rectangle([s[0]-0.00001, s[1]-0.00001,
      s[0]+0.00001, s[1]+0.00001], null, false);
    var zone = ee.Image.constant(1)
      .clip(seedBox.centroid(1).buffer(spec.qualityRadiusM))
      .unmask(0, false).toByte();
    water = water.and(zone).toByte();
  }
  var kernel = ee.Kernel.circle({radius: 300, units: 'meters'});
  var shore = water.focalMax({kernel: kernel})
    .and(water.focalMin({kernel: kernel}).not()).selfMask();
  var check = p.foot.updateMask(water.selfMask()).rename('s2_foot_lake')
    .addBands(p.lm.updateMask(shore).rename('l8_clear_shore'))
    .addBands(p.sm.updateMask(shore).rename('s2_clear_shore'))
    .addBands(p.joint.updateMask(shore).rename('joint_clear_shore'));
  return check.reduceRegion({reducer: ee.Reducer.mean(),
    geometry: p.box, scale: 30, maxPixels: 1e8, tileScale: 4});
}

function exportReplacement(target, row) {
  var p = products(target, row.s2);
  var stem = 'Q2_FAST_' + target.lake + '_' +
    target.l8.replace(/-/g, '') + '_' + row.s2.replace(/-/g, '');
  Export.image.toDrive({image: p.s2, description: stem + '_S2',
    folder: CFG.folder, region: p.box, crs: p.spec.crs,
    crsTransform: [10, 0, 0, 0, -10, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: 0}});
  Export.image.toDrive({image: p.masks, description: stem + '_QA_JRC',
    folder: CFG.folder, region: p.box, crs: p.spec.crs,
    crsTransform: [30, 0, 0, 0, -30, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}});
}

var panel = ui.Panel({style: {width: '470px', padding: '10px'}});
panel.add(ui.Label('Q2 S2 replacement quality screen',
  {fontWeight: 'bold', fontSize: '18px'}));
panel.add(ui.Label('Five failed pairings only. S2 is checked against the ' +
  'fixed JRC lake and 300 m shoreline zone, without viewing model output.'));
var labels = CFG.targets.map(function(t) {
  return t.lake + ' | L8 ' + t.l8 + ' | rejected S2 ' + t.rejectedS2;
});
var picker = ui.Select({items: labels, value: labels[0]});
panel.add(picker);
var status = ui.Label('Choose a target and click Scan alternatives.');
panel.add(status);
var out = ui.Panel(); panel.add(out);
ui.root.insert(0, panel);
var serial = 0;
var prepared = {};

function scan() {
  var run = ++serial;
  out.clear();
  var target = CFG.targets[labels.indexOf(picker.getValue())];
  var box = ee.Geometry.Rectangle(CFG.specs[target.lake].box, null, false);
  var day = ee.Date(target.l8);
  status.setValue('Listing S2 acquisition days within +/-4 days...');
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(box)
    .filterDate(day.advance(-CFG.gap, 'day'),
      day.advance(CFG.gap + 1, 'day'))
    .map(function(image) {
      return image.set('inventory_day',
        ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'));
    });
  collection.aggregate_array('inventory_day').distinct().sort()
    .evaluate(function(days, error) {
      if (run !== serial) return;
      if (error) { status.setValue('Date listing failed: ' + JSON.stringify(error)); return; }
      days = days.filter(function(d) { return d !== target.rejectedS2; });
      if (!days.length) {
        status.setValue('No alternative S2 day within four days. Replace L8 date.');
        return;
      }
      var rows = [];
      function next(i) {
        if (run !== serial) return;
        if (i === days.length) {
          rows.sort(function(a, b) { return b.joint - a.joint; });
          var eligible = rows.filter(function(r) { return r.pass; });
          status.setValue('Checked ' + rows.length + ' S2 dates sequentially; ' +
            eligible.length + ' passed. No predictions used.');
          rows.forEach(function(r) {
            out.add(ui.Label(r.s2 + ' | footprint ' +
              (100 * r.foot).toFixed(1) + '% | S2 shore ' +
              (100 * r.s2clear).toFixed(1) + '% | joint ' +
              (100 * r.joint).toFixed(1) + '% | ' +
              (r.pass ? 'PASS' : 'REJECT') +
              (r.error ? ' | GEE error: ' + r.error : '')));
          });
          if (!eligible.length) {
            out.add(ui.Label('STOP: replace this Landsat date; do not force ' +
              'a cloudy or partial S2 pairing.'));
            return;
          }
          var choices = eligible.map(function(r) {
            return r.s2 + ' | joint ' + (100 * r.joint).toFixed(1) + '%';
          });
          var select = ui.Select({items: choices, value: choices[0]});
          out.add(select);
          out.add(ui.Button('Prepare replacement S2 + QA: 2 tasks', function() {
            var row = eligible[choices.indexOf(select.getValue())];
            var key = target.lake + '_' + target.l8 + '_' + row.s2;
            if (prepared[key]) {
              status.setValue('This replacement was already prepared in this Run.');
              return;
            }
            prepared[key] = true;
            exportReplacement(target, row);
            status.setValue('Two replacement tasks prepared for ' + key +
              '. Start them in Tasks; keep the existing L8.');
            print('Q2_S2_REPLACEMENT_SELECTED', target, row);
          }));
          return;
        }
        status.setValue('Checking alternative ' + (i + 1) + '/' + days.length +
          ': ' + days[i] + ' (sequential, avoids concurrent aggregations)');
        qualityCheck(target, days[i]).evaluate(function(value, checkError) {
          if (run !== serial) return;
          if (checkError) {
            rows.push({s2: days[i], foot: 0, s2clear: 0,
              joint: 0, pass: false, error: JSON.stringify(checkError)});
          } else {
            var foot = Number(value.s2_foot_lake || 0);
            var s2clear = Number(value.s2_clear_shore || 0);
            var joint = Number(value.joint_clear_shore || 0);
            rows.push({s2: days[i], foot: foot, s2clear: s2clear,
              joint: joint,
              pass: foot >= 0.98 && s2clear >= 0.50 && joint >= 0.40});
          }
          next(i + 1);
        });
      }
      next(0);
    });
}

panel.insert(4, ui.Button('Scan alternatives', scan));
Map.setCenter(102.8, 24.6, 8);
