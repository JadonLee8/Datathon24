import React, {
  useState,
  useRef,
  LegacyRef,
  MutableRefObject,
  useEffect,
  useCallback,
} from "react";
import { MapContainer, TileLayer, Marker } from "react-leaflet";
import { Map } from "leaflet";
import "leaflet/dist/leaflet.css";
import dynamic from "next/dynamic";

export default function OpenStreetMapComponent({
  setRef,
}: {
  setRef: (ref: Map | null) => void;
}) {
  const mapRef = useRef<Map | null>(null);
  const [center, setCenter] = useState({
    lat: -4.043477,
    lng: 39.668205,
  });

  return (
    <div className="flex justify-center h-[500px]">
      <div className="basis-11/12 bg-lightmaroon ring-2 ring-white ring-offset-lightmaroon ring-offset-2 overflow-hidden rounded-sm">
        <MapContainer
          center={center}
          zoom={13}
          ref={setRef}
          className="h-full "
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        </MapContainer>
      </div>
    </div>
  );
}
