import React from 'react';
import { Link } from "react-router-dom";
import { NavLink } from 'react-router-dom';
import HistoryWrapper from './HistoryWrapper';
import SampleChoiceWrapper from './SampleChoiceWrapper';

const links = [
  { pathname: '/training', text: 'Training', icon: <SampleChoiceWrapper /> },
  { pathname: '/history', text: 'History', icon: <HistoryWrapper /> },
];

const Navbar: React.FC = () => {
  return (
    <div className="w-full">
      <div className="flex flex-row">
        <div className="my-auto bg-indigo-500 p-4 pl-4 pr-8 w-1/3">Learning From Human Preferences</div>
        <div className="w-full flex flex-row bg-gradient-to-r from-indigo-500 to-blue-600 py-2">
          {links.map((link) => (
            <NavLink to={link.pathname} className="my-auto p-2 mx-4 rounded-xl flex flex-col bg-blue-600 outline outlien-black outline-1 hover:bg-blue-700">{link.text}</NavLink>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Navbar;
