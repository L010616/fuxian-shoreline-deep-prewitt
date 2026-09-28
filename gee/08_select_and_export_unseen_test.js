// Fuxian Lake independent test expansion (2024-2025).
// Run once in Google Earth Engine. The script selects one previously unused
// Landsat 8 / Sentinel-2 pair in each four-month season of each year (6 dates).
// Selection is based only on cloud-free shoreline coverage and time gap. The
// best pair in every window is retained; its quality values are exported for QA.

var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var proj48 = ee.Projection('EPSG:32648');
var l8Transform = [30, 0, 277935, 0, -30, 2727015];
var s2Transform = [10, 0, 277960, 0, -10, 2726990];
var l8Box = ee.Geometry.Rectangle([277935, 2694165, 294195, 2727015], proj48, false);
var s2Box = ee.Geometry.Rectangle([277960, 2694180, 294190, 2726990], proj48, false);

var usedDates = ee.List([
  '2020-01-02', '2020-05-09', '2021-02-05', '2021-05-12',
  '2022-03-12', '2022-08-19', '2023-01-26', '2023-03-31',
  '2024-02-14', '2025-05-07', '2025-12-01'
]);

// The 2019 JRC lake extent is used only to define the search and QA zones.
var water2019 = ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first().select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox,
  scale: 30,
  geometryType: 'polygon',
  eightConnected: true,
  maxPixels: 1e7,
  geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);

function l8Valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0))
    .and(im.select('SR_B2').mask())
    .rename('valid').unmask(0);
}

function s2Valid(im) {
  var scl = im.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3)).and(scl.lt(8))
    .and(im.select('B2').mask()).rename('valid').unmask(0);
}

function meanOverShore(mask, scale) {
  return ee.Number(mask.unmask(0).rename('valid').reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: shore,
    scale: scale,
    maxPixels: 1e8,
    tileScale: 4
  }).get('valid'));
}

var s2All = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2023-12-30', '2026-01-03');

var l8All = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi)
  .filterDate('2024-01-01', '2026-01-01')
  .filter(ee.Filter.eq('WRS_PATH', 129))
  .filter(ee.Filter.eq('WRS_ROW', 43))
  .filter(ee.Filter.lt('CLOUD_COVER', 70));

var offsets = ee.List([-2, -1, 0, 1, 2]);

function bestS2Pair(l8Date) {
  var options = ee.FeatureCollection(offsets.map(function(offset) {
    offset = ee.Number(offset);
    var day = l8Date.advance(offset, 'day');
    var daily = s2All.filterDate(day, day.advance(1, 'day'));
    var count = daily.size();
    var clear = ee.Number(ee.Algorithms.If(
      count.gt(0),
      meanOverShore(daily.map(s2Valid).max(), 20),
      -1
    ));
    var score = clear.subtract(offset.abs().multiply(0.001));
    return ee.Feature(null, {
      s2_date: day.format('YYYY-MM-dd'),
      s2_count: count,
      s2_clear_shore: clear,
      day_gap: offset.abs(),
      signed_day_gap: offset,
      pair_score: score
    });
  }));
  return ee.Feature(options.sort('pair_score', false).first());
}

var l8List = l8All.toList(l8All.size());
var candidates = ee.FeatureCollection(l8List.map(function(obj) {
  var im = ee.Image(obj);
  var date = im.date();
  var dateText = date.format('YYYY-MM-dd');
  var pair = bestS2Pair(date);
  var l8Clear = meanOverShore(l8Valid(im), 30);
  var s2Clear = ee.Number(pair.get('s2_clear_shore'));
  var gap = ee.Number(pair.get('day_gap'));
  var score = l8Clear.multiply(0.55)
    .add(s2Clear.max(0).multiply(0.45))
    .subtract(gap.multiply(0.002));
  return ee.Feature(null, {
    l8_date: dateText,
    l8_index: im.get('system:index'),
    l8_product_id: im.get('LANDSAT_PRODUCT_ID'),
    l8_scene_cloud_pct: im.get('CLOUD_COVER'),
    l8_clear_shore: l8Clear,
    s2_date: pair.get('s2_date'),
    s2_scene_count: pair.get('s2_count'),
    s2_clear_shore: s2Clear,
    day_gap: gap,
    unused: ee.Number(ee.Algorithms.If(usedDates.contains(dateText), 0, 1)),
    quality_score: score
  });
})).filter(ee.Filter.eq('unused', 1))
  .filter(ee.Filter.gt('s2_scene_count', 0))
  .filter(ee.Filter.lte('day_gap', 2));

// One best pair in Jan-Apr, May-Aug and Sep-Dec for each year.
var windows = [
  ['2024-01-01', '2024-05-01', '2024_Jan-Apr'],
  ['2024-05-01', '2024-09-01', '2024_May-Aug'],
  ['2024-09-01', '2025-01-01', '2024_Sep-Dec'],
  ['2025-01-01', '2025-05-01', '2025_Jan-Apr'],
  ['2025-05-01', '2025-09-01', '2025_May-Aug'],
  ['2025-09-01', '2026-01-01', '2025_Sep-Dec']
];

var seasonalSelected = ee.FeatureCollection(windows.map(function(w) {
  var subset = candidates
    .filter(ee.Filter.gte('l8_date', w[0]))
    .filter(ee.Filter.lt('l8_date', w[1]))
    .sort('quality_score', false);
  var fallback = ee.Feature(null, {
    l8_date: 'NONE', s2_date: 'NONE', quality_score: -1
  });
  var best = ee.Feature(ee.Algorithms.If(subset.size().gt(0), subset.first(), fallback));
  return best.set({season_window: w[2], available: subset.size().gt(0)});
})).filter(ee.Filter.eq('available', 1));

// If a four-month window has no usable pair, fill the missing slot with the
// highest-quality unused pair from the same year. This guarantees three dates
// per year as long as that year contains at least three paired acquisitions.
function selectThreeForYear(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  var startText = start.format('YYYY-MM-dd');
  var endText = end.format('YYYY-MM-dd');
  var seasonal = seasonalSelected
    .filter(ee.Filter.gte('l8_date', startText))
    .filter(ee.Filter.lt('l8_date', endText));
  var chosenDates = seasonal.aggregate_array('l8_date');
  var remaining = candidates
    .filter(ee.Filter.gte('l8_date', startText))
    .filter(ee.Filter.lt('l8_date', endText))
    .map(function(f) {
      return f.set('_already_chosen', chosenDates.contains(f.get('l8_date')));
    })
    .filter(ee.Filter.eq('_already_chosen', false))
    .sort('quality_score', false)
    .map(function(f) {
      return f.set({season_window: year + '_annual_fallback', available: true});
    });
  var need = ee.Number(3).subtract(seasonal.size()).max(0);
  return seasonal.merge(remaining.limit(need));
}

var selected = selectThreeForYear(2024).merge(selectThreeForYear(2025));

print('ALL_QUALITY_CANDIDATES', candidates.sort('l8_date'));
print('SELECTED_UNSEEN_TEST_DATES', selected.sort('l8_date'));
print('SELECTED_COUNT_EXPECTED_6', selected.size());

Export.table.toDrive({
  collection: selected.sort('l8_date'),
  description: 'Fuxian_unseen_test_manifest_2024_2025_v2',
  folder: 'Fuxian_unseen_test_v2',
  fileFormat: 'CSV'
});

function buildRecord(rec) {
  var source = ee.Image('LANDSAT/LC08/C02/T1_L2/' + rec.l8_index);
  var reflectance = source
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    .multiply(0.0000275).add(-0.2)
    .rename(['B2', 'B3', 'B4', 'B5', 'B6', 'B7']);
  var lv = l8Valid(source);

  var s2Start = ee.Date(rec.s2_date);
  var s2col = s2All.filterDate(s2Start, s2Start.advance(1, 'day'))
    .map(function(im) { return im.addBands(s2Valid(im), null, true); });
  var s2m = s2col.qualityMosaic('valid');
  var sv = s2col.select('valid').max().unmask(0).rename('s2_valid');
  var joint = lv.and(sv).rename('joint_valid');

  var masks = ee.Image(0).byte().paint(shore, 1).rename('shore_zone')
    .addBands(lv.rename('l8_valid'))
    .addBands(sv)
    .addBands(joint)
    .addBands(ee.Image(0).byte().paint(roi, 1).rename('roi_zone'))
    .toByte().unmask(0).clip(l8Box);
  var l8 = reflectance.addBands(lv).toFloat().unmask(-9999).clip(l8Box);
  var s2 = s2m.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL', 'valid'])
    .toUint16().unmask(0).clip(s2Box);
  return {l8: l8, s2: s2, masks: masks};
}

function prepareExports(records, start, stop) {
  records.slice(start, stop).forEach(function(rec) {
    var item = buildRecord(rec);
    var l8tag = rec.l8_date.replace(/-/g, '');
    var s2tag = rec.s2_date.replace(/-/g, '');
    Export.image.toDrive({
      image: item.l8,
      description: 'Fuxian_unseen_v2_L8_' + l8tag,
      folder: 'Fuxian_unseen_test_v2',
      region: l8Box,
      crs: 'EPSG:32648',
      crsTransform: l8Transform,
      maxPixels: 1e8,
      fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true, noData: -9999}
    });
    Export.image.toDrive({
      image: item.s2,
      description: 'Fuxian_unseen_v2_S2_' + s2tag + '_for_' + l8tag,
      folder: 'Fuxian_unseen_test_v2',
      region: s2Box,
      crs: 'EPSG:32648',
      crsTransform: s2Transform,
      maxPixels: 1e8,
      fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true, noData: 0}
    });
    Export.image.toDrive({
      image: item.masks,
      description: 'Fuxian_unseen_v2_masks_' + l8tag,
      folder: 'Fuxian_unseen_test_v2',
      region: l8Box,
      crs: 'EPSG:32648',
      crsTransform: l8Transform,
      maxPixels: 1e8,
      fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true}
    });
  });
}

var panel = ui.Panel({style: {width: '420px', padding: '10px'}});
panel.add(ui.Label('Fuxian unseen-test export', {
  fontSize: '18px', fontWeight: 'bold'
}));
var status = ui.Label('Screening dates; please wait...');
panel.add(status);
ui.root.insert(0, panel);

selected.sort('l8_date').evaluate(function(fc, error) {
  if (error) {
    status.setValue('Screening failed: ' + error);
    return;
  }
  var records = fc.features.map(function(f) { return f.properties; });
  status.setValue('Selected ' + records.length + ' dates: ' +
    records.map(function(r) { return r.l8_date; }).join(', '));
  records.forEach(function(r) {
    panel.add(ui.Label(
      r.season_window + ': ' + r.l8_date + ' / ' + r.s2_date +
      ' | L8 ' + (100 * r.l8_clear_shore).toFixed(1) + '%' +
      ' | S2 ' + (100 * r.s2_clear_shore).toFixed(1) + '%',
      {fontSize: '11px'}
    ));
  });
  var firstCount = Math.min(3, records.length);
  var lastCount = Math.max(0, records.length - 3);
  panel.add(ui.Button('Prepare first ' + firstCount + ' dates (' +
    (firstCount * 3) + ' image tasks)', function() {
    prepareExports(records, 0, 3);
    status.setValue('Prepared ' + (firstCount * 3) +
      ' image tasks for the first group. Start them in Tasks.');
  }));
  panel.add(ui.Button('Prepare last ' + lastCount + ' dates (' +
    (lastCount * 3) + ' image tasks)', function() {
    prepareExports(records, 3, 6);
    status.setValue('Prepared ' + (lastCount * 3) +
      ' image tasks for the last group. Start them in Tasks.');
  }));
});

Map.centerObject(lake, 10);
Map.addLayer(lake, {color: '00A6D6'}, '2019 lake prior');
Map.addLayer(shore, {color: 'FF8C00'}, '300 m QA shore zone', false);
