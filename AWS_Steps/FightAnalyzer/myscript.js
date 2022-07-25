const de_ancient = ['SideEntrance', 'BombsiteA', 'TSpawn', 'CTSpawn', 'House', 'Tunnel', 'TSideUpper', 'Water', 'Outside', 'Middle', 'MainHall', 'Alley', 'Ruins', 'SideHall', 'TopofMid', 'TSideLower', 'Ramp', 'BombsiteB', 'BackHall', 'None']
const de_cache = ['Entrance', 'BombsiteA', 'LongA', 'BombsiteB', 'Garage', 'Middle', 'LongHall', 'Catwalk', 'TSpawn', 'Quad', 'Warehouse', 'Forklift', 'Dumpster', 'Window', 'Rafters', 'Back', 'CTSpawn', 'StorageRoom', 'ARamp', 'Truck', 'Squeaky', 'Roof', 'Ducts', 'None']
const de_cbble = ['BPlatform', 'Courtyard', 'BombsiteA', 'CTSpawn', 'BombsiteB', 'ARamp', 'LowerTunnel', 'Tunnels', 'Patio', 'Balcony', 'UpperTunnel', 'LongA', 'Hut', 'Underpass', 'Connector', 'TunnelStairs', 'TRamp', 'Catwalk', 'Hay', 'TSpawn', 'SideDoor', 'TMain', 'SnipersNest', 'None']
const de_dust2 = ['BombsiteA', 'LongDoors', 'BombsiteB', 'BDoors', 'OutsideTunnel', 'OutsideLong', 'ExtendedA', 'Hole', 'Catwalk', 'Side', 'TSpawn', 'UpperTunnel', 'Short', 'TRamp', 'Middle', 'UnderA', 'ARamp', 'LongA', 'TunnelStairs', 'TopofMid', 'ShortStairs', 'CTSpawn', 'Pit', 'Ramp', 'MidDoors', 'LowerTunnel', 'None']
const de_inferno = ['CTSpawn', 'BombsiteA', 'TRamp', 'TSpawn', 'LowerMid', 'TopofMid', 'Quad', 'Upstairs', 'BombsiteB', 'Banana', 'Ruins', 'Middle', 'BackAlley', 'Apartments', 'Graveyard', 'SecondMid', 'Balcony', 'Pit', 'Arch', 'Bridge', 'Underpass', 'Library', 'Deck', 'Kitchen', 'None']
const de_mirage = ['CTSpawn', 'BombsiteA', 'TRamp', 'TicketBooth', 'TopofMid', 'Shop', 'Apartments', 'BombsiteB', 'PalaceAlley', 'TSpawn', 'Catwalk', 'House', 'SnipersNest', 'Jungle', 'Tunnel', 'BackAlley', 'Middle', 'PalaceInterior', 'TunnelStairs', 'Ladder', 'Stairs', 'SideAlley', 'Scaffolding', 'Truck', 'Connector', 'Balcony', 'None']
const de_nuke = ['Hut', 'Roof', 'Outside', 'BombsiteB', 'Ramp', 'TSpawn', 'BombsiteA', 'Decon', 'Silo', 'Lobby', 'Vending', 'Garage', 'CTSpawn', 'Squeaky', 'Vents', 'Rafters', 'HutRoof', 'Tunnels', 'LockerRoom', 'Observation', 'Secret', 'Heaven', 'Hell', 'Mini', 'Admin', 'Catwalk', 'Crane', 'Trophy', 'Control', 'None']
const de_overpass = ['BackofA', 'LowerPark', 'BombsiteA', 'Connector', 'Canal', 'Tunnels', 'Fountain', 'UpperPark', 'Walkway', 'Water', 'StorageRoom', 'SnipersNest', 'Lobby', 'Construction', 'Playground', 'Restroom', 'Pipe', 'TSpawn', 'Alley', 'UnderA', 'BombsiteB', 'TStairs', 'SideAlley', 'Bridge', 'Pit', 'Stairs', 'None']
const de_train = ['Tunnel2', 'TStairs', 'BombsiteA', 'TMain', 'Ivy', 'Kitchen', 'BombsiteB', 'CTSpawn', 'Tunnel', 'LadderTop', 'BackofB', 'LockerRoom', 'Connector', 'TSpawn', 'Tunnel1', 'LadderBottom', 'SnipersNest', 'Scaffolding', 'Alley', 'Dumpster', 'BPlatform', 'Tunnels', 'PopDog', 'ElectricalBox', 'None']
const de_vertigo = ['Pit', 'Side', 'BombsiteA', 'BombsiteB', 'Tunnels', 'ARamp', 'Elevator', 'TSpawn', 'BackDoor', 'BackofA', 'Mid', 'LadderTop', 'Bridge', 'APlatform', 'Scaffolding', 'LadderBottom', 'CTSpawn', 'BackofB', 'TCorridorUp', 'TopofMid', 'Window', 'Crane', 'None']

const kill_weapons = ["CZ75 Auto", "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18",
    "P2000", "P250", "R8 Revolver", "Tec-9", "USP-S", "MAG-7", "Nova", "Sawed-Off", "XM1014", "M249",
    "Negev", "MAC-10", "MP5-SD", "MP7", "MP9", "P90", "PP-Bizon", "UMP-45", "AK-47", "AUG", "FAMAS",
    "Galil AR", "M4A1", "M4A4", "SG 553", "AWP", "G3SG1", "SCAR-20", "SSG 08", "Smoke Grenade", "Flashbang", "HE Grenade", "Incendiary Grenade", "Molotov", "Decoy Grenade", "Knife", "Zeus x27"]
const t_weapons_allowed = kill_weapons
const t_weapons_forbidden = kill_weapons
const ct_weapons_allowed = kill_weapons
const ct_weapons_forbidden = kill_weapons
const kill_classes = ["Pistols", "Heavy", "SMG", "Rifle", "Grenade", "Equipment"]
const t_classes_allowed = kill_classes
const t_classes_forbidden = kill_classes
const ct_classes_allowed = kill_classes
const ct_classes_forbidden = kill_classes
const maps_map = new Map();
maps_map.set("de_ancient", de_ancient)
maps_map.set("de_cache", de_cache)
maps_map.set("de_cbble", de_cbble)
maps_map.set("de_dust2", de_dust2)
maps_map.set("de_inferno", de_inferno)
maps_map.set("de_mirage", de_mirage)
maps_map.set("de_nuke", de_nuke)
maps_map.set("de_overpass", de_overpass)
maps_map.set("de_train", de_train)
maps_map.set("de_vertigo", de_vertigo)
const weapons_map = new Map();
weapons_map.set('Kill_Weapon', kill_weapons);
weapons_map.set('Kill_Classes', kill_classes);
weapons_map.set('CT_WeaponAllowed', ct_weapons_allowed);
weapons_map.set('CT_ClassesAllowed', ct_classes_allowed);
weapons_map.set('CT_WeaponForbidden', ct_weapons_forbidden);
weapons_map.set('CT_ClassesForbidden', ct_classes_forbidden);
weapons_map.set('T_WeaponAllowed', t_weapons_allowed);
weapons_map.set('T_WeaponForbidden', t_weapons_forbidden);
weapons_map.set('T_ClassesAllowed', t_classes_allowed);
weapons_map.set('T_ClassesForbidden', t_classes_forbidden);

function on_load() {
    for (const [key, value] of maps_map) {
        map_div = document.createElement("div")
        map_div.setAttribute("id", key + "_div")
        for (const side of ["CT", "T"]) {
            side_div = document.createElement("div")
            side_div.setAttribute("id", key + "_" + side + "_div");
            dropdown = document.createElement("div");
            dropdown.setAttribute("class", "dropdown");
            dropdown.setAttribute("id", key + "_" + side + "_drop");
            dropdown.style.display = "inline-block"
            dropbtn = document.createElement("button");
            dropbtn.setAttribute("class", "dropbtn");
            dropbtn.innerHTML = side + "_Positions";
            dropdown.appendChild(dropbtn);
            dropdown_content = document.createElement("div");
            dropdown_content.setAttribute("class", "dropdown-content");
            dropdown_content.setAttribute("id", key + "_" + side);
            dropdown.appendChild(dropdown_content)
            map_select = document.getElementById("map_select")
            for (let i = 0; i < value.length; i++) {
                var addBtn = document.createElement("button");
                addBtn.innerHTML = value[i];
                addBtn.onclick = AddElement;
                dropdown_content.appendChild(addBtn)
            }

            side_div.appendChild(dropdown)
            list = document.createElement("ul")
            list.setAttribute("id", key + "_" + side + "_list")
            list.style.display = "inline-block"
            side_div.style.display = "none"
            side_div.appendChild(list)
            map_div.appendChild(side_div)
        }
        component = document.getElementById("map_component")
        component.appendChild(map_div)
        option = document.createElement("option")
        option.setAttribute("value", key)
        option.setAttribute("id", key + "_select")
        option.innerHTML = key
        if (key == "de_dust2") {
            option.setAttribute("selected", "selected")
        }
        map_select.appendChild(option)
    }
    for (const [key, value] of weapons_map) {
        div = document.createElement("div")
        div.setAttribute("id", key + "_div");
        dropdown = document.createElement("div");
        dropdown.setAttribute("class", "dropdown");
        dropdown.setAttribute("id", key + "_drop");
        dropdown.style.display = "inline-block"
        dropbtn = document.createElement("button");
        dropbtn.setAttribute("class", "dropbtn");
        dropbtn.innerHTML = key.split(/(?=[A-Z])/).slice(-1);
        dropdown.appendChild(dropbtn);
        dropdown_content = document.createElement("div");
        dropdown_content.setAttribute("class", "dropdown-content");
        dropdown_content.setAttribute("id", key);
        dropdown.appendChild(dropdown_content)
        for (let i = 0; i < value.length; i++) {
            var addBtn = document.createElement("button");
            addBtn.innerHTML = value[i];
            addBtn.onclick = AddElement;
            dropdown_content.appendChild(addBtn)
        }
        var component_name = key.split("_")[0].toLowerCase() + "_component"
        component = document.getElementById(component_name)
        div.appendChild(dropdown)
        list = document.createElement("ul")
        list.setAttribute("id", key + "_list")
        list.style.display = "inline-block"
        div.appendChild(list)
        component.appendChild(div)
    }
    document.getElementById("CT_Classes").click();
    document.getElementById("Kill_Classes").click();
    document.getElementById("T_Classes").click();
    document.getElementById("map_select").onchange();
}

var currentKillValue = 0;
function select_kill_weapon_classes(myRadio) {
    currentKillValue = myRadio.value;
    target = myRadio.id.split("_")[0]
    if (currentKillValue == "classes") {

        document.getElementById(target + "_Weapon_div").style.display = "none";
        document.getElementById(target + "_Classes_div").style.display = "block";

    }
    else {

        document.getElementById(target + "_Weapon_div").style.display = "block";
        document.getElementById(target + "_Classes_div").style.display = "none";

    }
}
var currentTValue = 0;
function select_T_weapon_classes(myRadio) {
    currentTValue = myRadio.value;
    target = myRadio.id.split("_")[0]
    if (currentTValue == "classes") {
        for (const permission of ["Allowed", "Forbidden"]) {
            document.getElementById(target + "_Weapon" + permission + "_div").style.display = "none";
            document.getElementById(target + "_Classes" + permission + "_div").style.display = "block";
        }
    }
    else {
        for (const permission of ["Allowed", "Forbidden"]) {
            document.getElementById(target + "_Weapon" + permission + "_div").style.display = "block";
            document.getElementById(target + "_Classes" + permission + "_div").style.display = "none";
        }
    }
}

var currentCTValue = 0;
function select_CT_weapon_classes(myRadio) {
    currentCTValue = myRadio.value;
    target = myRadio.id.split("_")[0]
    if (currentCTValue == "classes") {
        for (const permission of ["Allowed", "Forbidden"]) {
            document.getElementById(target + "_Weapon" + permission + "_div").style.display = "none";
            document.getElementById(target + "_Classes" + permission + "_div").style.display = "block";
        }
    }
    else {
        for (const permission of ["Allowed", "Forbidden"]) {
            document.getElementById(target + "_Weapon" + permission + "_div").style.display = "block";
            document.getElementById(target + "_Classes" + permission + "_div").style.display = "none";
        }
    }
}

var old_value = "0";
function select_map(mySelect) {
    currentValue = mySelect.value
    map_hull = document.getElementById("map_hull")
    map_hull.src = "map_hulls/hulls_" + currentValue + ".png"
    for (const [key, value] of maps_map) {
        for (const side of ["CT", "T"]) {
            if (key == old_value) {
                document.getElementById(key + "_" + side + "_div").style.display = "none";
            }
            if (key == currentValue) {
                document.getElementById(key + "_" + side + "_div").style.display = "block";
            }
        }
    }
    old_value = currentValue
}


function removeElement(e) {
    var addBtn = document.createElement("button");
    addBtn.innerHTML = e.target.innerHTML;
    addBtn.onclick = AddElement;
    element_id = e.target.parentNode.parentNode.id
    div = element_id.slice(0, element_id.lastIndexOf('_'))
    document.getElementById(div).appendChild(addBtn);
    e.target.parentNode.remove();
}

function AddElement(e) {
    var l = document.createElement("li");
    var removeBtn = document.createElement("button");
    removeBtn.setAttribute("class", "dynamic_button");
    removeBtn.innerHTML = e.target.innerHTML;
    removeBtn.onclick = removeElement;
    l.appendChild(removeBtn);
    category = e.target.parentNode.id
    document.getElementById(category + "_list").appendChild(l);
    e.target.remove();
}

function adjust_end_min(start_number) {
    var value = start_number.value
    if (parseInt(value) < parseInt(start_number.min) || value == "") {
        value = start_number.min
        start_number.setAttribute("value", value)
        start_number.value = value
    }
    else if (parseInt(value) > parseInt(start_number.max)) {
        value = start_number.max
        start_number.setAttribute("value", value)
        start_number.value = value
    }
    end_input = document.getElementById("end_second")
    end_input.setAttribute("min", parseInt(value) + 1)
}

function adjust_start_max(end_number) {
    var value = end_number.value
    if (parseInt(value) > parseInt(end_number.max)) {
        value = end_number.max
        end_number.setAttribute("value", value)
        end_number.value = value
    }
    else if (parseInt(value) < parseInt(end_number.min) || value == "") {
        value = end_number.min
        end_number.setAttribute("value", value)
        end_number.value = value
    }
    start_input = document.getElementById("start_second")
    start_input.setAttribute("max", parseInt(value) - 1)
}

function fill_list_with_children_text(my_list_id) {
    const my_list = []
    children = document.getElementById(my_list_id).children
    for (var i = 0; i < children.length; i++) {
        my_list.push(children[i].innerText)
    }
    return my_list
}

// selecting loading div

async function collect_query_input() {
    var map_name = document.getElementById("map_select").value
    const CT_pos = fill_list_with_children_text(map_name + "_CT_list")
    const T_pos = fill_list_with_children_text(map_name + "_T_list")
    var use_weapons_classes_Kill = document.querySelector('input[name="Kill_weapons_classes"]:checked').value
    var Kill_classes_q = []
    var Kill_weapons_q = []
    if (use_weapons_classes_Kill == "weapons") {
        Kill_weapons_q = fill_list_with_children_text("Kill_Weapon_list")
    }
    else {
        Kill_classes_q = fill_list_with_children_text("Kill_Classes_list")
    }
    var CT_classes_forbidden_q = []
    var CT_classes_allowed_q = []
    var CT_weapons_forbidden_q = []
    var CT_weapons_allowed_q = []
    use_weapons_classes_CT = document.querySelector('input[name="CT_weapons_classes"]:checked').value
    if (use_weapons_classes_CT == "weapons") {
        CT_weapons_forbidden_q = fill_list_with_children_text("CT_WeaponForbidden_list")
        CT_weapons_allowed_q = fill_list_with_children_text("CT_WeaponAllowed_list")
    }
    else {
        CT_classes_forbidden_q = fill_list_with_children_text("CT_ClassesForbidden_list")
        CT_classes_allowed_q = fill_list_with_children_text("CT_ClassesAllowed_list")
    }
    var T_classes_forbidden_q = []
    var T_classes_allowed_q = []
    var T_weapons_forbidden_q = []
    var T_weapons_allowed_q = []
    use_weapons_classes_T = document.querySelector('input[name="T_weapons_classes"]:checked').value
    if (use_weapons_classes_T == "weapons") {
        T_weapons_forbidden_q = fill_list_with_children_text("T_WeaponForbidden_list")
        T_weapons_allowed_q = fill_list_with_children_text("T_WeaponAllowed_list")
    }
    else {
        T_classes_forbidden_q = fill_list_with_children_text("T_ClassesForbidden_list")
        T_classes_allowed_q = fill_list_with_children_text("T_ClassesAllowed_list")
    }
    const times = []
    times.push(document.getElementById("start_second").value)
    times.push(document.getElementById("end_second").value)
    result_text = document.getElementById("result_text")
    result_text.innerHTML = "Retrieving information. Please wait."
    event_data = {
        "map_name": map_name,
        "weapons": { "Kill": Kill_weapons_q, "T": { "Allowed": T_weapons_allowed_q, "Forbidden": T_weapons_forbidden_q }, "CT": { "Allowed": CT_weapons_allowed_q, "Forbidden": CT_weapons_forbidden_q } },
        "classes": { "Kill": Kill_classes_q, "T": { "Allowed": T_classes_allowed_q, "Forbidden": T_classes_forbidden_q }, "CT": { "Allowed": CT_classes_allowed_q, "Forbidden": CT_classes_forbidden_q } },
        "positions": { "CT": CT_pos, "T": T_pos },
        "use_weapons_classes": { "CT": use_weapons_classes_CT, "T": use_weapons_classes_T, "Kill": use_weapons_classes_Kill },
        "times": { "start": times[0], "end": times[1] }
    }
    // instantiate a headers object
    var myHeaders = new Headers();
    // add content type header to object
    myHeaders.append("Content-Type", "application/json");
    // using built in JSON utility package turn object to string and store in a variable
    var raw = JSON.stringify(event_data);
    var requestOptions = {
        method: 'POST',
        headers: myHeaders,
        body: raw,
        redirect: 'follow'
    };
    var date = new Date();
    var h = date.getHours();
    var m = date.getMinutes();
    var s = date.getSeconds();
    var date = new Date();
    displayLoading(date)
    // make API call with parameters and use promises to get response
    const result = await call_API("https://uq7f1xuyn1.execute-api.eu-central-1.amazonaws.com/dev", requestOptions)
    hideLoading()
    try {
        body = JSON.parse(result.body)
        status_code = result.statusCode
        if (status_code == 200) {
            result_text.innerHTML = "A total of " + body.Situations_found + " situations matching your description have been found.<br>Out of those the CT's won " + body.CT_win_percentage + "%."
        }
        else if (status_code == 500) {
            result_text.innerHTML = "An error occured while processing your request: " + body.errorMessage
        }
        else if (status_code == 408) {
            result_text.innerHTML = "The request timed out with the error message: " + body.errorMessage + "!<br>Your selection is probably too broad. Try a narrower one!"
        }
        else {
            result_text.innerHTML = "An unkown status_code of " + status_code + " was returned from your query. I do not know what happend here..."
        }
    }
    catch {
        result_text.innerHTML = "Got an invalid response. It is likely that the gateway timed out.<br>Your selection is probably too broad. Try a narrower one!"
    }

}

async function call_API(url, requestOptions) {
    return fetch(url, requestOptions).then((res, reject) => {
        const rejectResponse = {
            "error_type": "SERVER_ERROR",
            "error": true
        }
        if (res.ok === true) { return res.json() }
        else { return reject(rejectReponse) }
    }).catch(error => console.log('error', error));
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


// showing loading
function displayLoading(date) {
    const loader = document.getElementById("loading")
    const timer = document.getElementById("timer")
    loader.style.display = "inline-block";
    timer.style.display = "inline"
    updateTime(date, timer)
}

function updateTime(oldDate, timer) {
    if (timer.style.display != "none") {
        var newDate = new Date();
        setTimeout(updateTime, 1000, oldDate, timer);
        timer.innerHTML = msToHMS(newDate - oldDate);
    }
}

function msToHMS(ms) {
    // 1- Convert to seconds:
    var seconds = ms / 1000;

    // 3- Extract minutes:
    var minutes = parseInt(seconds / 60); // 60 seconds in 1 minute

    // 4- Keep only seconds not extracted to minutes:
    seconds = parseInt(seconds % 60);

    // 5 - Format so it shows a leading zero if needed
    let minutesStr = ("00" + minutes).slice(-2);
    let secondsStr = ("00" + seconds).slice(-2);

    return minutesStr + "m:" + secondsStr + "s"
}

// hiding loading
function hideLoading() {
    const loader = document.getElementById("loading")
    loader.style.display = "none";
    timer.style.display = "none"
}