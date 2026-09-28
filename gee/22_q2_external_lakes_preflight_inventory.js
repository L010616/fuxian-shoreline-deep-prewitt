// Fuxian Lake Q2 extension: external-lake candidate inventory with preflight.
// Paste the whole script into https://code.earthengine.google.com/ and click Run.
// Select one lake -> List L8 -> select one date -> Find S2 -> select S2 ->
// preview or prepare three GeoTIFF export tasks in the Tasks tab.
// This is prediction-blind, metadata-only candidate discovery. Scene cloud
// percentages are NOT shoreline clear fractions. Review exported images before
// freezing dates or using any model prediction. No reduceToVectors/reduceRegion
// runs during discovery, avoiding "Too many concurrent aggregations". One
// Lake and shore QA aggregation runs only when an export is requested.
// This version rejects clearly incomplete S2 tiles and cloudy L8 shorelines
// before preparing any download tasks. Local 12-sector QC remains mandatory.

var CONFIG = {
  years: [2020, 2021, 2022, 2023, 2024, 2025],
  l8PerYearSeason: 3,
  maxPairGapDays: 4,
  driveFolder: 'Fuxian_Q2_external_candidates',
  lakes: {
    Dianchi_Waihai: {
      box: [102.57, 24.62, 102.87, 25.06], crs: 'EPSG:32648',
      centroid: [102.68659, 24.82303],
      // Three points measured on the main 2019 JRC lake component.
      anchors: [[102.67310, 24.92293], [102.71056, 24.82385],
        [102.64620, 24.71237]]
    },
    Erhai: {
      box: [100.04, 25.56, 100.35, 26.02], crs: 'EPSG:32647',
      anchors: [[100.17, 25.96], [100.19, 25.80], [100.25, 25.64]]
    },
    Xingyun: {
      box: [102.69, 24.25, 102.88, 24.43], crs: 'EPSG:32648',
      anchors: [[102.78, 24.39], [102.78, 24.34], [102.79, 24.29]]
    }
  }
};

function coverageAnchored(collection, spec) {
  // filterBounds operates on the scene footprint, with no pixel reduction.
  // Requiring north/middle/south lake anchors before sorting by scene cloud
  // avoids selecting a clear adjacent WRS row that clips only a lake tip.
  spec.anchors.forEach(function(point) {
    collection = collection.filterBounds(ee.Geometry.Point(point));
  });
  return collection;
}

function l8Valid(image) {
  return image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0))
    .and(image.select('SR_B2').mask())
    .rename('l8_valid').unmask(0).toByte();
}

function s2Valid(image) {
  var scl = image.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.lt(8)).and(image.select('B2').mask())
    .rename('s2_valid').unmask(0).toByte();
}

// A lake can intersect two neighbouring Landsat WRS rows on the same day.
// Never export just the single scene shown in the metadata shortlist.
function l8SourceForDay(record, box) {
  return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(box)
    .filterDate(record.l8_date, ee.Date(record.l8_date).advance(1, 'day'))
    .sort('CLOUD_COVER', false).mosaic();
}

function pairPreflight(record, pair, box) {
  var source = l8SourceForDay(record, box);
  var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(box)
    .filterDate(pair.s2_date, ee.Date(pair.s2_date).advance(1, 'day'));
  var water = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
    .filter(ee.Filter.eq('year', 2019)).first())
    .select('waterClass').gte(2).unmask(0, false);
  var shore = water.focalMax({radius: 300, units: 'meters'})
    .and(water.focalMin({radius: 300, units: 'meters'}).not());
  // sameFootprint=false counts uncovered lake pixels as 0.
  var l8Foot = source.select('SR_B2').mask().unmask(0, false)
    .rename('l8_foot').toByte();
  var s2Foot = s2Collection.select('B2').count().gt(0)
    .unmask(0, false).rename('s2_foot').toByte();
  var l8Clear = l8Valid(source).unmask(0, false);
  var s2Clear = s2Collection.map(s2Valid).max()
    .unmask(0, false).rename('s2_valid').toByte();
  var lakeStats = l8Foot.addBands(s2Foot).updateMask(water)
    .reduceRegion({
      reducer: ee.Reducer.mean(), geometry: box, scale: 30,
      maxPixels: 1e7, tileScale: 8
    });
  var shoreStats = l8Clear.addBands(s2Clear)
    .addBands(l8Clear.and(s2Clear).rename('joint_valid').toByte())
    .updateMask(shore).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: box, scale: 30,
      maxPixels: 1e7, tileScale: 8
    });
  // A prediction-blind approximation of the local 12-sector JRC boundary
  // check. The fixed centroid is measured on the 2019 Dianchi lake component.
  // The local raster QC still makes the final decision after download.
  var centroid = CONFIG.lakes[record.lake].centroid;
  if (!centroid) return ee.Dictionary(lakeStats)
    .combine(ee.Dictionary(shoreStats), true);
  var ll = ee.Image.pixelLonLat();
  var dx = ll.select('longitude').subtract(centroid[0])
    .multiply(Math.cos(centroid[1] * Math.PI / 180));
  var dy = ll.select('latitude').subtract(centroid[1]);
  var sector = dy.atan2(dx).add(Math.PI).divide(2 * Math.PI)
    .multiply(12).floor().toInt().mod(12).rename('sector');
  var boundary = water.and(water.focalMin({
    radius: 1, kernelType: 'square', units: 'pixels'
  }).not())
    .selfMask();
  var sectorStats = l8Clear.and(s2Clear).rename('joint_valid')
    .addBands(sector).updateMask(boundary).reduceRegion({
      reducer: ee.Reducer.mean().group({groupField: 1, groupName: 'sector'}),
      geometry: box, scale: 30, maxPixels: 1e7, tileScale: 8
    });
  return ee.Dictionary(lakeStats).combine(ee.Dictionary(shoreStats), true)
    .combine(ee.Dictionary(sectorStats), true);
}

var lakeNames = ['Dianchi_Waihai', 'Erhai', 'Xingyun'];
var panel = ui.Panel({style: {width: '450px', padding: '10px'}});
panel.add(ui.Label('External-lake FAST inventory',
  {fontSize: '18px', fontWeight: 'bold'}));
panel.add(ui.Label('Metadata shortlist only. Export preflight checks both ' +
  'sensor footprints and 300-m shore validity. Local 12-sector QC still applies.'));
var lakePicker = ui.Select({items: lakeNames, value: lakeNames[0]});
panel.add(lakePicker);
var status = ui.Label('Choose a lake and click List Landsat dates.');
panel.add(status);
var results = ui.Panel();
panel.add(results);
ui.root.insert(0, panel);
var requestSerial = 0;
var prepared = {};

function listLandsat() {
  var serial = ++requestSerial;
  results.clear();
  var lakeKey = lakePicker.getValue();
  var spec = CONFIG.lakes[lakeKey];
  var box = ee.Geometry.Rectangle(spec.box, null, false);
  status.setValue('Listing Landsat scene metadata for ' + lakeKey + '...');
  var all = ee.ImageCollection([]);
  CONFIG.years.forEach(function(year) {
    var annual = coverageAnchored(
      ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterBounds(box)
      .filterDate(String(year) + '-01-01', String(year + 1) + '-01-01')
      .filter(ee.Filter.lt('CLOUD_COVER', 90)), spec);
    var dry = annual.filter(ee.Filter.or(
      ee.Filter.calendarRange(1, 4, 'month'),
      ee.Filter.calendarRange(11, 12, 'month')))
      .sort('CLOUD_COVER').limit(CONFIG.l8PerYearSeason)
      .map(function(image) { return image.set('inventory_season', 'dry'); });
    var wet = annual.filter(ee.Filter.calendarRange(6, 10, 'month'))
      .sort('CLOUD_COVER').limit(CONFIG.l8PerYearSeason)
      .map(function(image) { return image.set('inventory_season', 'wet'); });
    all = all.merge(dry).merge(wet);
  });
  var rows = ee.FeatureCollection(all.toList(36).map(function(item) {
    var image = ee.Image(item);
    return ee.Feature(null, {
      lake: lakeKey,
      season: image.get('inventory_season'),
      l8_date: ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
      l8_id: image.get('system:id'),
      l8_cloud_scene_pct: image.get('CLOUD_COVER'),
      status: 'UNREVIEWED_CANDIDATE'
    });
  }));
  rows.evaluate(function(fc, error) {
    if (serial !== requestSerial) return;
    if (error) {
      status.setValue('Landsat listing failed: ' + JSON.stringify(error));
      return;
    }
    var records = fc.features.map(function(f) { return f.properties; });
    var seen = {};
    records = records.filter(function(r) {
      if (seen[r.l8_date]) return false;
      seen[r.l8_date] = true;
      return true;
    });
    records.sort(function(a, b) {
      if (a.season !== b.season) return a.season < b.season ? -1 : 1;
      return Number(a.l8_cloud_scene_pct) - Number(b.l8_cloud_scene_pct);
    });
    status.setValue(lakeKey + ': ' + records.length +
      ' Landsat candidates listed. Pick one, then find its S2 pair.');
    if (!records.length) return;

    results.add(ui.Button('Prepare Landsat shortlist CSV (1 task)', function() {
      Export.table.toDrive({collection: ee.FeatureCollection(records.map(
        function(r) { return ee.Feature(null, r); })),
        description: 'Q2_FAST_' + lakeKey + '_L8_shortlist',
        folder: CONFIG.driveFolder, fileFormat: 'CSV'});
      status.setValue('One CSV task prepared. Open the Tasks tab to start it.');
    }));

    var l8Labels = records.map(function(r) {
      return r.l8_date + ' [' + r.season + '] scene cloud ' +
        Number(r.l8_cloud_scene_pct).toFixed(1) + '%';
    });
    var l8Picker = ui.Select({items: l8Labels, value: l8Labels[0]});
    results.add(ui.Label('Landsat date')); results.add(l8Picker);
    var s2Panel = ui.Panel(); results.add(s2Panel);
    results.add(ui.Button('Find S2 within +/- 4 days', function() {
      s2Panel.clear();
      var record = records[l8Labels.indexOf(l8Picker.getValue())];
      var day = ee.Date(record.l8_date);
      status.setValue('Finding Sentinel-2 dates near ' + record.l8_date + '...');
      var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(box)
        .filterDate(day.advance(-CONFIG.maxPairGapDays, 'day'),
          day.advance(CONFIG.maxPairGapDays + 1, 'day'))
        .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30);
      var s2Rows = ee.FeatureCollection(collection.toList(30).map(function(item) {
        var image = ee.Image(item);
        var s2Date = ee.Date(image.get('system:time_start'));
        return ee.Feature(null, {
          s2_date: s2Date.format('YYYY-MM-dd'),
          s2_cloud_tile_pct: image.get('CLOUDY_PIXEL_PERCENTAGE'),
          gap_days: s2Date.difference(day, 'day').abs().round(),
          s2_id: image.get('system:id')
        });
      }));
      s2Rows.evaluate(function(s2fc, s2error) {
        if (serial !== requestSerial) return;
        if (s2error) {
          status.setValue('S2 listing failed: ' + JSON.stringify(s2error));
          return;
        }
        var pairs = s2fc.features.map(function(f) { return f.properties; });
        var seenDays = {};
        pairs = pairs.filter(function(p) {
          if (seenDays[p.s2_date]) return false;
          seenDays[p.s2_date] = true;
          return true;
        });
        pairs.sort(function(a, b) {
          return Number(a.s2_cloud_tile_pct) - Number(b.s2_cloud_tile_pct);
        });
        if (!pairs.length) {
          status.setValue('No S2 scene within four days of ' + record.l8_date +
            '. Select another Landsat date.');
          return;
        }
        status.setValue(pairs.length + ' S2 dates found for L8 ' +
          record.l8_date + '. Select and inspect one.');
        var s2Labels = pairs.map(function(p) {
          return p.s2_date + ' (gap ' + p.gap_days + ' d, tile cloud ' +
            Number(p.s2_cloud_tile_pct).toFixed(1) + '%)';
        });
        var s2Picker = ui.Select({items: s2Labels, value: s2Labels[0]});
        s2Panel.add(ui.Label('Sentinel-2 date'));
        s2Panel.add(s2Picker);
        function chosen() {
          return pairs[s2Labels.indexOf(s2Picker.getValue())];
        }
        s2Panel.add(ui.Button('Preview selected pair', function() {
          var pair = chosen();
          var images = buildImages(record, pair, box);
          Map.layers().reset();
          Map.addLayer(images.l8, {bands: ['B4', 'B3', 'B2'],
            min: 0, max: 0.25}, 'L8 ' + record.l8_date, true);
          Map.addLayer(images.s2, {bands: ['B4', 'B3', 'B2'],
            min: 0, max: 3000}, 'S2 ' + pair.s2_date, false);
          Map.centerObject(box, 10);
        }));
        function preparePair(includeS2) {
          var pair = chosen();
          var taskKey = lakeKey + '_' + record.l8_date + '_' + pair.s2_date;
          if (prepared[taskKey]) {
            status.setValue('This pair was already prepared in this Run.');
            return;
          }
          status.setValue('Checking L8/S2 lake coverage and shore validity for ' +
            record.l8_date + ' / ' + pair.s2_date + '...');
          pairPreflight(record, pair, box).evaluate(function(check, checkError) {
            if (serial !== requestSerial) return;
            if (checkError) {
              status.setValue('Coverage check failed: ' +
                JSON.stringify(checkError));
              return;
            }
            var l8Foot = Number(check.l8_foot || 0);
            var s2Foot = Number(check.s2_foot || 0);
            var l8Shore = Number(check.l8_valid || 0);
            var s2Shore = Number(check.s2_valid || 0);
            var jointShore = Number(check.joint_valid || 0);
            if (l8Foot < 0.98 || s2Foot < 0.98 || l8Shore < 0.50 ||
                s2Shore < 0.50 || jointShore < 0.40) {
              status.setValue('STOP: lake footprint L8/S2 ' +
                (100 * l8Foot).toFixed(1) + '/' + (100 * s2Foot).toFixed(1) +
                '%, shore valid L8/S2/joint ' +
                (100 * l8Shore).toFixed(1) + '/' +
                (100 * s2Shore).toFixed(1) + '/' +
                (100 * jointShore).toFixed(1) +
                '%. Need >=98/98% footprint, >=50/50/40% shore. Pick another pair.');
              return;
            }
            var sectorGroups = check.groups || [];
            var coveredSectors = sectorGroups.filter(function(group) {
              return Number(group.mean || 0) >= 0.50;
            }).length;
            if (lakeKey === 'Dianchi_Waihai' && coveredSectors < 8) {
              status.setValue('STOP: only ' + coveredSectors +
                '/12 approximate shoreline sectors are >=50% jointly valid. ' +
                'Need at least 8/12; pick another date.');
              return;
            }
            prepared[taskKey] = true;
            exportPair(record, pair, box, spec, includeS2);
            status.setValue((includeS2 ? 'Three' : 'Two') +
              ' tasks prepared. L8/S2 lake footprint ' +
              (100 * l8Foot).toFixed(1) + '/' + (100 * s2Foot).toFixed(1) +
              '%, joint shore valid ' + (100 * jointShore).toFixed(1) +
              '%, approximate sectors ' + coveredSectors +
              '/12. Start in Tasks; local 12-sector QC is still required.');
          });
        }
        s2Panel.add(ui.Button('Check coverage + export 3 GeoTIFFs',
          function() { preparePair(true); }));
        s2Panel.add(ui.Button('Check coverage + re-export L8 and QA only',
          function() { preparePair(false); }));
      });
    }));
  });
}

function buildImages(record, pair, box) {
  var source = l8SourceForDay(record, box);
  var lm = l8Valid(source);
  var l8 = source.select(['SR_B2', 'SR_B3', 'SR_B4',
      'SR_B5', 'SR_B6', 'SR_B7'])
    .multiply(0.0000275).add(-0.2)
    .rename(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    .addBands(lm.rename('valid'))
    .toFloat().unmask(-9999).clip(box);
  var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(box)
    .filterDate(pair.s2_date, ee.Date(pair.s2_date).advance(1, 'day'))
    .map(function(image) { return image.addBands(s2Valid(image)); });
  var sm = s2Collection.select('s2_valid').max()
    .unmask(0).rename('s2_valid').toByte();
  var s2 = s2Collection.qualityMosaic('s2_valid')
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL', 's2_valid'])
    .toUint16().unmask(0).clip(box);
  var jrc = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
    .filter(ee.Filter.eq('year', 2019)).first())
    .select('waterClass').unmask(0).toByte().rename('jrc2019_class');
  var masks = jrc.addBands(lm).addBands(sm)
    .addBands(lm.and(sm).rename('joint_valid').toByte())
    .toByte().unmask(0).clip(box);
  return {l8: l8, s2: s2, masks: masks};
}

function exportPair(record, pair, box, spec, includeS2) {
  var images = buildImages(record, pair, box);
  var stem = 'Q2_FAST_' + record.lake + '_' +
    record.l8_date.replace(/-/g, '') + '_' + pair.s2_date.replace(/-/g, '');
  Export.image.toDrive({image: images.l8, description: stem + '_L8',
    folder: CONFIG.driveFolder, region: box, crs: spec.crs,
    crsTransform: [30, 0, 0, 0, -30, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: -9999}});
  if (includeS2) {
    Export.image.toDrive({image: images.s2, description: stem + '_S2',
      folder: CONFIG.driveFolder, region: box, crs: spec.crs,
      crsTransform: [10, 0, 0, 0, -10, 0], maxPixels: 1e9,
      fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true, noData: 0}});
  }
  Export.image.toDrive({image: images.masks, description: stem + '_QA_JRC',
    folder: CONFIG.driveFolder, region: box, crs: spec.crs,
    crsTransform: [30, 0, 0, 0, -30, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}});
}

panel.insert(4, ui.Button('List Landsat dates', listLandsat));
Map.setCenter(102.8, 24.6, 8);
