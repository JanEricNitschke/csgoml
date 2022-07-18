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

const attacker_weapons = ["CZ75 Auto", "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18",
    "P2000", "P250", "R8 Revolver", "Tec-9", "USP-S", "MAG-7", "Nova", "Sawed-Off", "XM1014", "M249",
    "Negev", "MAC-10", "MP5-SD", "MP7", "MP9", "P90", "PP-Bizon", "UMP-45", "AK-47", "AUG", "FAMAS",
    "Galil AR", "M4A1", "M4A4", "SG 553", "AWP", "G3SG1", "SCAR-20", "SSG 08"]
const victim_weapons_allowed = attacker_weapons.concat(["Smoke Grenade", "Flashbang", "HE Grenade", "Incendiary Grenade", "Molotov", "Decoy Grenade", "Knife", "Zeus x27"])
const victim_weapons_forbidden = attacker_weapons.concat(["Smoke Grenade", "Flashbang", "HE Grenade", "Incendiary Grenade", "Molotov", "Decoy Grenade", "Knife", "Zeus x27"])
const attacker_classes = ["Pistols", "Heavy", "SMG", "Rifle", "Grenade", "Equipment"]
const victim_classes_allowed = ["Pistols", "Heavy", "SMG", "Rifle"]
const victim_classes_forbidden = ["Pistols", "Heavy", "SMG", "Rifle"]
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
weapons_map.set('AttackerWeapon', attacker_weapons);
weapons_map.set('AttackerClasses', attacker_classes);
weapons_map.set('VictimWeaponAllowed', victim_weapons_allowed);
weapons_map.set('VictimWeaponForbidden', victim_weapons_forbidden);
weapons_map.set('VictimClassesAllowed', victim_classes_allowed);
weapons_map.set('VictimClassesForbidden', victim_classes_forbidden);

function on_load() {
    for (const [key, value] of maps_map) {
        for (const side of ["CT", "T"]) {
            div = document.createElement("div")
            div.setAttribute("id", key + "_" + side + "_div");
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
            column = document.getElementById("map_column")
            div.appendChild(dropdown)
            list = document.createElement("ul")
            list.setAttribute("id", key + "_" + side + "_list")
            list.style.display = "inline-block"
            div.style.display = "none"
            div.appendChild(list)
            column.appendChild(div)
        }
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
        var column_name = key.split(/(?=[A-Z])/)[0].toLowerCase() + "_column"
        column = document.getElementById(column_name)
        div.appendChild(dropdown)
        list = document.createElement("ul")
        list.setAttribute("id", key + "_list")
        list.style.display = "inline-block"
        div.appendChild(list)
        column.appendChild(div)
    }
    document.getElementById("Attacker_Classes").click();
    document.getElementById("Victim_Classes").click();
    document.getElementById("map_select").onchange();
}

var currentAttackerValue = 0;
function select_attacker_weapon_classes(myRadio) {
    currentAttackerValue = myRadio.value;
    target = myRadio.id.split("_")[0]
    if (currentAttackerValue == "Classes") {

        document.getElementById(target + "Weapon_div").style.display = "none";
        document.getElementById(target + "Classes_div").style.display = "block";

    }
    else {

        document.getElementById(target + "Weapon_div").style.display = "block";
        document.getElementById(target + "Classes_div").style.display = "none";

    }
}
var currentVictimValue = 0;
function select_victim_weapon_classes(myRadio) {
    currentVictimValue = myRadio.value;
    target = myRadio.id.split("_")[0]
    if (currentVictimValue == "Classes") {
        for (const permission of ["Allowed", "Forbidden"]) {
            document.getElementById(target + "Weapon" + permission + "_div").style.display = "none";
            document.getElementById(target + "Classes" + permission + "_div").style.display = "block";
        }
    }
    else {
        for (const permission of ["Allowed", "Forbidden"]) {
            document.getElementById(target + "Weapon" + permission + "_div").style.display = "block";
            document.getElementById(target + "Classes" + permission + "_div").style.display = "none";
        }
    }
}

var old_value = "0";
function select_map(mySelect) {
    currentValue = mySelect.value
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